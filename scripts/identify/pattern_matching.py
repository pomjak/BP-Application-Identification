"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/03/2025
"""

from prefixspan import prefixspan
from database import Database
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from logger import Logger
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# from spade import spade as sp

import constants as col_names


class PatternMatchingMethod:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

        self.first_guess = 0
        self.second_guess = 0
        self.third_guess = 0

        self.uniqueness = 0
        self.usage_of_patterns = {}
        self.used_once_count = [0, 0, 0]
        self.used_twice_count = [0, 0, 0]
        self.used_more_times = [0, 0, 0]
        self.used_never = [0, 0, 0]

    def display_statistics(self):
        print("________________________________________________________")
        print("Pattern matching:")
        print(f"Correct: {self.correct}")
        print(f"Incorrect: {self.incorrect}")
        total = self.correct + self.incorrect
        print(f"Total: {total}")
        print()
        print(
            f"First guess: {self.first_guess} ({round(self.first_guess / self.correct, 2)})"
        )
        print(
            f"Second guess: {self.second_guess} ({round(self.second_guess / self.correct, 2)})"
        )
        print(
            f"Third guess: {self.third_guess} ({round(self.third_guess / self.correct, 2)})"
        )
        print()
        print(f"Accuracy 1st guess : {round(self.first_guess / total, 4)}")
        print(f"Accuracy 2nd guess : {round(self.second_guess / (total), 4)}")
        print(f"Accuracy 3rd guess : {round(self.third_guess / (total), 4)}")
        print(f"Accuracy overall: {round(self.correct / (total), 4)}")
        print(f"Error rate: {round(self.incorrect / (total), 4)}")
        print()

        print(f"Number of unique patterns sets: {self.uniq_count}")
        print("Usage of distinct sets:")
        print(f"{'Usage':<20}{'1st Guess':<12}{'2nd Guess':<12}{'3rd Guess':<12}")

        categories = [
            ("Used only once", self.used_once_count),
            ("Used at least twice", self.used_twice_count),
            ("Used more times", self.used_more_times),
            ("Never used", self.used_never),
        ]

        for label, data in categories:
            total = sum(data) if sum(data) > 0 else 1
            percentages = [f"{(x / total):.4f}" for x in data]
            print(
                f"{label:<20}{data[0]:<3}({percentages[0]:<5}) {data[1]:<3}({percentages[1]:<5}) {data[2]:<3}({percentages[2]:<5})"
            )

        print(f"Uniqueness: {round(self.used_once_count[0] / self.uniq_count, 4)}")

    def identify(self, df):
        raise NotImplementedError("This method should be overridden by subclasses")


class Apriori(PatternMatchingMethod):
    def train(self, db: Database):
        with Logger() as logger:
            logger.info("Training Apriori algorithm ...")
            data = db.get_train_df()
            groups = data.groupby(col_names.FILE)

            for _, group in groups:
                self._train_group(group, db)

    def _log_patterns(self, db):
        with Logger() as logger:
            logger.debug("Frequent patterns found: \n")
            for app in db.frequent_patterns:
                logger.debug(f"app: {app}")
                for file in db.frequent_patterns[app]:
                    logger.debug(f"filename: {file}")
                    logger.debug(f"patterns: {db.frequent_patterns[app][file]}\n")

    def _init_db_for_app(self, app, db):
        if app in db.frequent_patterns:
            return
        else:
            db.frequent_patterns[app] = {}
            with Logger() as logger:
                logger.debug(f"Creating new entry for {app}")

    def _add_patterns_to_db(self, app, launch, patterns, db):
        with Logger() as logger:
            for existing_launch in db.frequent_patterns[app]:
                existing_df = db.frequent_patterns[app][existing_launch]

                if existing_df.equals(pd.DataFrame(patterns)):  # Found a duplicate
                    logger.debug(f"Found duplicate for {app}")
                    break  # Stop checking further.

            else:
                # If no duplicate was found, add the new pattern.
                db.frequent_patterns[app][launch] = pd.DataFrame(patterns)

                logger.debug(f"Found {len(patterns)} frequent item sets for {app} \n")

    def _train_group(self, group, db):
        with Logger() as logger:
            app_name = group[col_names.APP_NAME].iloc[0]
            launch = group[col_names.FILE].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            frequent_item_sets = self._execute_apriori(group)

            self._init_db_for_app(app_name, db)

            # If a duplicate is found, log it and stop checking.
            self._add_patterns_to_db(app_name, launch, frequent_item_sets, db)

    def _preprocess(self, data):
        data = data.drop(
            columns=[
                col_names.FILE,
                col_names.APP_NAME,
                # col_names.JA4_S,  # Too ambiguous, ignoring.
            ]
        )
        data = data.astype(str)

        # Serialize the data.
        data_list = data.values.tolist()

        # Transform into one-hot encoding.
        te = TransactionEncoder()
        te_ary = te.fit(data_list).transform(data_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Convert to boolean values to ensure that the apriori algorithm works correctly,
        # as it requires the input data to be in a binary format.
        df_encoded = df.astype(bool)

        return df_encoded

    def _execute_apriori(self, group):
        processed_group = self._preprocess(group)

        freq_items_set = apriori(
            processed_group, min_support=0.01, use_colnames=True, max_len=3
        )
        return freq_items_set

    # 0.8427
    def _jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def fast_weighted_jaccard(df1, df2):
        df1_dict = dict(zip(df1["itemsets"].apply(frozenset), df1["support"]))
        df2_dict = dict(zip(df2["itemsets"].apply(frozenset), df2["support"]))

        # Compute intersection & union first to reduce dictionary lookups
        intersection_keys = df1_dict.keys() & df2_dict.keys()
        union_keys = df1_dict.keys() | df2_dict.keys()

        min_support_sum = sum(min(df1_dict[k], df2_dict[k]) for k in intersection_keys)
        max_support_sum = sum(
            max(df1_dict.get(k, 0), df2_dict.get(k, 0)) for k in union_keys
        )

        return min_support_sum / max_support_sum if max_support_sum else 0

    # 0.8315
    def _cosine_similarity(self, set1, set2):
        # Handle empty sets
        if not set1 or not set2:
            return 0.0

        vectorizer = CountVectorizer(
            tokenizer=lambda x: x, lowercase=False, token_pattern=None
        )
        try:
            matrix = vectorizer.fit_transform([list(set1), list(set2)])
            return cosine_similarity(matrix)[0, 1]
        except ValueError:
            return 0.0

    # 0.8427
    def _dice_coefficient(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return (
            (2 * intersection) / (len(set1) + len(set2))
            if (len(set1) + len(set2)) != 0
            else 0
        )

    # 0.7528
    def _overlap_coefficient(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return (
            intersection / min(len(set1), len(set2))
            if min(len(set1), len(set2)) != 0
            else 0
        )

    # 0.8427
    def _tversky_index(self, set1, set2, alpha=0.5, beta=0.5):
        intersection = len(set1.intersection(set2))
        only_in_set1 = len(set1 - set2)
        only_in_set2 = len(set2 - set1)
        return (
            intersection / (intersection + alpha * only_in_set1 + beta * only_in_set2)
            if (intersection + alpha * only_in_set1 + beta * only_in_set2) != 0
            else 0
        )

    def _calculate_similarity(self, found_items_sets, db):
        similarities = []

        # Compute threshold using mode, handle empty mode case
        mode_values = found_items_sets["support"].mode()
        threshold = mode_values[0] if not mode_values.empty else 0.0

        # Split DataFrame into low and high support sets
        found_low_support = found_items_sets[found_items_sets["support"] < threshold]
        found_high_support = found_items_sets[found_items_sets["support"] >= threshold]

        # Convert itemsets to frozenset for fast Jaccard computation
        found_low_support_set = frozenset(found_low_support["itemsets"])
        found_high_support_set = frozenset(found_high_support["itemsets"])

        # Precompute total support and weight fractions
        total_support = found_items_sets["support"].sum()
        high_support_weight = found_high_support["support"].sum() / total_support
        low_support_weight = found_low_support["support"].sum() / total_support
        cross_support_weight = (high_support_weight + low_support_weight) / 2

        for app, launches in db.frequent_patterns.items():
            for launch, train_items_sets in launches.items():
                # Compute threshold for training set
                mode_values = train_items_sets["support"].mode()
                threshold = mode_values[0] if not mode_values.empty else 0.0

                # Split training set into low and high support
                train_low_support = train_items_sets[
                    train_items_sets["support"] < threshold
                ]
                train_high_support = train_items_sets[
                    train_items_sets["support"] >= threshold
                ]

                # Convert to frozensets
                train_low_support_set = frozenset(train_low_support["itemsets"])
                train_high_support_set = frozenset(train_high_support["itemsets"])

                # Compute Jaccard similarities
                low_support_sim = self._jaccard_similarity(
                    found_low_support_set, train_low_support_set
                )
                high_support_sim = self._jaccard_similarity(
                    found_high_support_set, train_high_support_set
                )
                cross_support_sim = self._jaccard_similarity(
                    found_low_support_set, train_high_support_set
                )
                cross_support_sim2 = self._jaccard_similarity(
                    found_high_support_set, train_low_support_set
                )

                # Compute final weighted similarity score
                similarity = (
                    high_support_sim * high_support_weight
                    + low_support_sim * low_support_weight
                    + cross_support_sim * cross_support_weight
                    + cross_support_sim2 * cross_support_weight
                )

                # Maintain a min-heap of size 3 for top similarities
                if len(similarities) < 3:
                    heapq.heappush(similarities, (similarity, app, launch))
                else:
                    heapq.heappushpop(similarities, (similarity, app, launch))

        # Return top 3 highest similarities
        return heapq.nlargest(3, similarities)

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using Apriori algorithm ...")
            # Retrieve test data and group it by app.
            test_ds = db.get_test_df()
            groups = test_ds.groupby(col_names.FILE)
            results = {}

            for _, group in groups:
                # Find frequent patterns for each group (one app).
                found_items_sets = self._execute_apriori(group)
                # Retrieve top 3 most similar apps.
                top_similarities = self._calculate_similarity(found_items_sets, db)

                real_app = group[col_names.APP_NAME].iloc[0]
                logger.info(
                    f" real app: {real_app}, top 3 similarities: {[(round(sim, 2), app, file) for sim, app, file in top_similarities]}"
                )
                # Update statistics based on the results.
                self._update_statistics(
                    real_app, top_similarities, db.frequent_patterns
                )
                for index, row in group.iterrows():
                    if index not in results:
                        results[index] = []
                    for sim in top_similarities:
                        results[index].append(sim[1])

            # Retrieve number of unique patterns sets in the database.
            self.uniq_count = self._get_number_of_unique_patterns_sets(
                db.frequent_patterns
            )
            # Count how often each set is used and its corresponding guess position.
            self._count_usage(db)
            db.context_results = results

    def _count_usage(self, db):
        for app, launches in db.frequent_patterns.items():
            for launch in launches:
                # Iterate over top 3 guesses and update stats.
                for pos in range(1, 4):
                    match self._get_usage_of_set(app, launch, pos):
                        case 0:
                            self.used_never[pos - 1] += 1
                        case 1:
                            self.used_once_count[pos - 1] += 1
                        case 2:
                            self.used_twice_count[pos - 1] += 1
                        case default:  # noqa: F841
                            self.used_more_times[pos - 1] += 1

    def _update_statistics(self, real_app, top_similarities, set_of_patterns):
        if top_similarities:
            self._check_top_guesses(real_app, top_similarities, set_of_patterns)
        else:
            self._log_no_similar_apps_found()

    def _check_top_guesses(self, real_app, top_similarities, set_of_patterns):
        # If real app is 1st guess, update stats. Else check 2nd and 3rd guess.
        if real_app == top_similarities[0][1]:
            self._update_correct_guess(1, set_of_patterns, top_similarities[0])

        elif len(top_similarities) > 1 and real_app == top_similarities[1][1]:
            self._update_correct_guess(2, set_of_patterns, top_similarities[1])

        elif len(top_similarities) > 2 and real_app == top_similarities[2][1]:
            self._update_correct_guess(3, set_of_patterns, top_similarities[2])

        else:
            self.incorrect += 1

    def _update_correct_guess(self, guess_rank, set_of_patterns, similarity):
        # Update stats based on which guess was correct.
        match guess_rank:
            case 1:
                self.first_guess += 1
                self._mark_set_as_used(similarity[1], similarity[2], 1)

            case 2:
                self.second_guess += 1
                self._mark_set_as_used(similarity[1], similarity[2], 2)

            case 3:
                self.third_guess += 1
                self._mark_set_as_used(similarity[1], similarity[2], 3)

        self.correct += 1

    def _log_no_similar_apps_found(self):
        with Logger() as logger:
            logger.warn("No similar apps found")

    def _mark_set_as_used(self, app, file, pos=1):
        """
        Mark set of patterns in training db as used for identifying the app correctly.
        """

        if app not in self.usage_of_patterns:
            self.usage_of_patterns[app] = {}
        if file not in self.usage_of_patterns[app]:
            self.usage_of_patterns[app][file] = [0, 0, 0]

        self.usage_of_patterns[app][file][pos - 1] += 1

    def _get_usage_of_set(self, app, file, pos=1):
        if app in self.usage_of_patterns and file in self.usage_of_patterns[app]:
            return self.usage_of_patterns[app][file][pos - 1]
        else:
            return 0

    def _get_number_of_unique_patterns_sets(self, trained_patterns, app=None):
        """
        Returns the number of unique patterns sets in the database if app is None,
        otherwise returns the number of unique patterns sets for the given app.
        """
        uniq = 0
        if app:
            return len(trained_patterns[app])
        else:
            for app in trained_patterns:
                # Db has already only distinct sets of patterns for each app.
                uniq += len(trained_patterns[app])
        return uniq

    def _log_usage_of_every_launch(self):
        with Logger() as logger:
            logger.debug("Usage of patterns:")
            count = 0
            for app in self.usage_of_patterns:
                logger.debug(f"App: {app}")
                for file in self.usage_of_patterns[app]:
                    count += 1
                    logger.debug(f"File: {file}")
                    logger.debug(f"Usage: {self.usage_of_patterns[app][file]}")
                    logger.debug(count)
                    logger.debug("\n")

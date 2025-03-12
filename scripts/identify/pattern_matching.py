"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 12/03/2025
"""

from prefixspan import prefixspan
from database import Database
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from logger import Logger

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
        print(f"{'Usage':<20}{'1st Guess':<15}{'2nd Guess':<15}{'3rd Guess':<15}")

        categories = [
            ("Used only once", self.used_once_count),
            ("Used at least twice", self.used_twice_count),
            ("Used more times", self.used_more_times),
            ("Never used", self.used_never),
        ]

        for label, data in categories:
            print(f"{label:<20}{data[0]:<15}{data[1]:<15}{data[2]:<15}")

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

    def _train_group(self, group, db):
        with Logger() as logger:
            app_name = group[col_names.APP_NAME].iloc[0]
            launch = group[col_names.FILE].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            frequent_item_sets = self.execute_apriori(group)

            if app_name not in db.frequent_patterns:
                db.frequent_patterns[app_name] = {}
                logger.debug(f"Creating new entry for {app_name}")

            # If a duplicate is found, log it and stop checking.
            for existing_launch in db.frequent_patterns[app_name]:
                existing_df = db.frequent_patterns[app_name][existing_launch]

                if existing_df.equals(
                    pd.DataFrame(frequent_item_sets)
                ):  # Found a duplicate
                    logger.debug(f"Found duplicate for {app_name}")
                    break  # Stop checking further

            else:
                # If no duplicate was found, add the new pattern.
                db.frequent_patterns[app_name][launch] = pd.DataFrame(
                    frequent_item_sets
                )

                logger.debug(
                    f"Found {len(frequent_item_sets)} frequent item sets for {app_name} \n"
                )

    def _preprocess(self, data):
        data = data.drop(
            columns=[
                col_names.FILE,
                col_names.APP_NAME,
                # col_names.JA4_S,  # Too ambiguous, worse for 1st match accuracy but better for overall accuracy.
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

    def execute_apriori(self, group):
        processed_group = self._preprocess(group)

        freq_items_set = apriori(
            processed_group, min_support=0.01, use_colnames=True, max_len=3
        )
        return freq_items_set

    # 0.82
    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # 0.82
    def dice_coefficient(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return (
            (2 * intersection) / (len(set1) + len(set2))
            if (len(set1) + len(set2)) != 0
            else 0
        )

    # 0.76
    def overlap_coefficient(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return (
            intersection / min(len(set1), len(set2))
            if min(len(set1), len(set2)) != 0
            else 0
        )

    # 0.82 0.8673
    def tversky_index(self, set1, set2, alpha=0.75, beta=0.25):
        intersection = len(set1.intersection(set2))
        only_in_set1 = len(set1 - set2)
        only_in_set2 = len(set2 - set1)
        return (
            intersection / (intersection + alpha * only_in_set1 + beta * only_in_set2)
            if (intersection + alpha * only_in_set1 + beta * only_in_set2) != 0
            else 0
        )

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using Apriori algorithm ...")

            test_ds = db.get_test_df()
            groups = test_ds.groupby(col_names.FILE)
            for _, group in groups:
                found_items_sets = self.execute_apriori(group)

                # check if the found frequent item sets are in the training dataset
                found_items_set = set(found_items_sets["itemsets"])
                similarities = []

                for app in db.frequent_patterns:
                    for filename in db.frequent_patterns[app]:
                        trained_items_set = set(
                            db.frequent_patterns[app][filename]["itemsets"]
                        )

                        similarity = self.jaccard_similarity(
                            found_items_set, trained_items_set
                        )
                        similarities.append((similarity, app, filename))

                # Sort similarities in descending order and get the top 3.
                top_similarities = sorted(similarities, reverse=True)[:3]
                real_app = group[col_names.APP_NAME].iloc[0]

                logger.info(
                    f" real app: {real_app}, top 3 similarities: {[(round(sim, 2), app, file) for sim, app, file in top_similarities]}"
                )

                self._update_statistics(
                    real_app, top_similarities, db.frequent_patterns
                )

            self._log_patterns(db)
            self.uniq_count = self.get_number_of_unique_patterns_sets(
                db.frequent_patterns
            )
            self.count_usage(db)

    def count_usage(self, db):
        for app in db.frequent_patterns:
            for file in db.frequent_patterns[app]:
                # iterate over 1 to 3 guesses
                for pos in range(1, 4):
                    if self._get_usage_of_set(app, file, pos) == 1:
                        self.used_once_count[pos - 1] += 1

                    elif self._get_usage_of_set(app, file, pos) == 2:
                        self.used_twice_count[pos - 1] += 1

                    elif self._get_usage_of_set(app, file, pos) == 0:
                        self.used_never[pos - 1] += 1

                    elif self._get_usage_of_set(app, file, pos) > 2:
                        self.used_more_times[pos - 1] += 1
                    else:
                        raise ValueError("Invalid usage count. Shouldn't happen.")

    def _update_statistics(self, real_app, top_similarities, set_of_patterns):
        if top_similarities:
            self._check_top_guesses(real_app, top_similarities, set_of_patterns)
        else:
            self._log_no_similar_apps_found()

    def _check_top_guesses(self, real_app, top_similarities, set_of_patterns):
        if real_app == top_similarities[0][1]:
            self._update_correct_guess(1, set_of_patterns, top_similarities[0])

        elif len(top_similarities) > 1 and real_app == top_similarities[1][1]:
            self._update_correct_guess(2, set_of_patterns, top_similarities[1])

        elif len(top_similarities) > 2 and real_app == top_similarities[2][1]:
            self._update_correct_guess(3, set_of_patterns, top_similarities[2])

        else:
            self.incorrect += 1

    def _update_correct_guess(self, guess_rank, set_of_patterns, similarity):
        if guess_rank == 1:
            self._mark_set_as_used(similarity[1], similarity[2], 1)
            self.first_guess += 1

        elif guess_rank == 2:
            self.second_guess += 1
            self._mark_set_as_used(similarity[1], similarity[2], 2)

        elif guess_rank == 3:
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

    def get_number_of_unique_patterns_sets(self, trained_patterns, app=None):
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


class PrefixSpan(PatternMatchingMethod):
    def __init__(self):
        pass

    def train(self, db):
        pass

    def identify(self, db):
        pass


class SPADE(PatternMatchingMethod):
    def __init__(self):
        pass

    def train(self, db):
        pass

    def identify(self, db):
        pass

"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 20/03/2025
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
        """
        Train the Apriori algorithm on dataset grouped by app for multiple launches,
        so that the frequent patterns are found over more launches.
        """
        with Logger() as logger:
            logger.info("Training Apriori algorithm ...")
            # Retrieve training data.
            data = db.get_train_df()
            # Group tls entries by app name
            tls_entries_of_apps = data.groupby(col_names.APP_NAME)

            # Train for each app with multiple launches and look for frequent patterns over more launches
            for _, multiple_launches_of_one_app in tls_entries_of_apps:
                self._train_group(multiple_launches_of_one_app, db)

    def log_patterns(self, db):
        with Logger() as logger:
            logger.debug("Frequent patterns found: \n")
            for app in db.frequent_patterns:
                logger.debug(f"app: {app}")
                logger.debug(f"patterns: {db.frequent_patterns[app]}\n")

    def _init_db_for_app(self, app, db):
        if app in db.frequent_patterns:
            return
        else:
            db.frequent_patterns[app] = pd.DataFrame()
            with Logger() as logger:
                logger.debug(f"Creating new entry for {app}")

    def _add_patterns_to_db(self, app, patterns, db):
        """
        Add only UNIQUE frequent patterns to DB.
        """
        db.frequent_patterns[app] = pd.DataFrame(patterns)
        db.frequent_patterns[app] = self._normalize_support(db.frequent_patterns[app])
        with Logger() as logger:
            logger.debug(f"Found {len(patterns)} frequent item sets for {app} \n")

    def _normalize_support(self, patterns_df):
        support_values = patterns_df["support"].values
        percentiles = np.percentile(
            support_values, np.arange(0, 101, 1)
        )  # Percentiles from 0 to 100

        # Map each support value to its percentile rank
        normalized_support = (
            np.searchsorted(percentiles, support_values, side="right") / 100.0
        )

        # Replace the original support values with the normalized values
        patterns_df["normalized_support"] = normalized_support

        return patterns_df

    def _train_group(self, group, db):
        with Logger() as logger:
            app_name = group[col_names.APP_NAME].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            frequent_item_sets = self._execute_apriori(group)
            self._init_db_for_app(app_name, db)

            self._add_patterns_to_db(app_name, frequent_item_sets, db)

    def _preprocess(self, data):
        data = data.drop(
            columns=[
                col_names.FILE,
                col_names.APP_NAME,
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

        freq_items_set = apriori(processed_group, min_support=0.01, use_colnames=True)
        return freq_items_set

    def _jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def _overlap_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return intersection / len(set1) if len(set1) != 0 else 0

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using Apriori algorithm ...")
            # Retrieve test data and group it by app.
            test_ds = db.get_test_df()
            test_ds_launches = test_ds.groupby(col_names.FILE)

            for _, launch in test_ds_launches:
                real_app = launch[col_names.APP_NAME].iloc[0]
                # Find similarity of tle entries in db of frequent patterns.
                top_guesses = self.find_similarity(db.frequent_patterns, launch)
                print(f"{real_app}:{top_guesses}")
                # Update statistics based on the results.
                self._update_statistics(real_app, top_guesses, db.frequent_patterns)

            # Retrieve number of unique patterns sets in the database.
            self.uniq_count = self._get_number_of_unique_patterns_sets(
                db.frequent_patterns
            )
            # Count how often each set is used and its corresponding guess position.
            self._count_usage(db)

    def find_similarity(self, frequent_patterns, tls_group):
        top_scores = {}
        stripped_tls = tls_group.drop(columns=[col_names.FILE, col_names.APP_NAME])
        tls_set = set(stripped_tls.values.flatten())

        for app, patterns in frequent_patterns.items():
            score = sum(
                self._jaccard_similarity(tls_set, set(row["itemsets"]))
                * row["normalized_support"]
                for _, row in patterns.iterrows()
            )
            if score > 0:
                top_scores[app] = score

        # Return top 3 scores with app name and scores
        return heapq.nlargest(3, top_scores.items(), key=lambda x: x[1])

    def _count_usage(self, db):
        for app in db.frequent_patterns:
            # Iterate over top 3 guesses and update stats.
            for pos in range(1, 4):
                match self._get_usage_of_set(app, pos):
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
        for rank, (app, _) in enumerate(top_similarities[:3], start=1):
            if real_app == app:
                self._update_correct_guess(rank, app)
                break
        else:
            self.incorrect += 1

    def _update_correct_guess(self, guess_rank, app):
        # Update stats based on which guess was correct.
        match guess_rank:
            case 1:
                self.first_guess += 1
                self._mark_set_as_used(app, 1)

            case 2:
                self.second_guess += 1
                self._mark_set_as_used(app, 2)

            case 3:
                self.third_guess += 1
                self._mark_set_as_used(app, 3)

        self.correct += 1

    def _log_no_similar_apps_found(self):
        with Logger() as logger:
            logger.warn("No similar apps found")

    def _mark_set_as_used(self, app, pos=1):
        """
        Mark set of patterns in training db as used for identifying the app correctly.
        """

        if app not in self.usage_of_patterns:
            self.usage_of_patterns[app] = [0, 0, 0]
        self.usage_of_patterns[app][pos - 1] += 1

    def _get_usage_of_set(self, app, pos=1):
        if app in self.usage_of_patterns:
            return self.usage_of_patterns[app][pos - 1]
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
                count += 1
                logger.debug(f"App: {app}")
                logger.debug(f"Usage: {self.usage_of_patterns[app]}")
                logger.debug(count)
                logger.debug("\n")

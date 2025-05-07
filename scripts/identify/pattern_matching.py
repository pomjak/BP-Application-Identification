"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 07/05/2025

CITATIONS OF SOURCES:
[1] CHOUDHARY G. A Beginner’s Guide to Apriori .... [Online]. Best Tech Blog For Programming .., 2. září 2023.
    Revidováno 8.10.2023. Dostupné z: https://programmerblog.net/a-beginners-guide-to-apriori-algorithm-in-python/.
    [cit. 2025-04-21].Path: Home; Python; A Beginner’s Guide to Apriori Algorithm in Python.
"""

from .database import Database
from .logger import Logger
import config

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from math import log
import operator
import csv
import os


class PatternMatchingMethod:
    def __init__(self, min_sup, version, max_candidates_size, csv_file=None):
        self.csv_file = csv_file
        self.min_support = min_sup
        self.ja_version = version
        self.candidate_size = max_candidates_size

        self.correct = [0] * self.candidate_size
        self.incorrect = 0

        self.comb_correct = [0] * self.candidate_size
        self.comb_incorrect = 0

        self.empty_candidates = 0
        self.empty_comb_candidates = 0

        self.len_of_candidates = []
        self.comb_len_of_candidates = []

        self.number_of_tls = 0
        self.pure_context = 0
        self.pure_context_comb = 0

        self.empty_ja = 0
        self.empty_ja_comb = 0

        self.context_using_whole_db = 0
        self.context_using_whole_db_comb = 0

        self.pattern_sim = {}

    def _update_statistics(self, real_app, top_similarities, is_comb=False):
        if top_similarities:
            self._check_top_guesses(real_app, top_similarities, is_comb)
        else:
            if is_comb:
                self.empty_comb_candidates += 1
            else:
                self.empty_candidates += 1
            self._log_no_similar_apps_found(real_app)

    def _check_top_guesses(self, real_app, top_similarities, is_comb=False):
        # If real app is 1st to candidate_size guess, update the statistics,
        # else increment incorrect.
        for rank, (app, _) in enumerate(
            top_similarities[: self.candidate_size], start=1
        ):
            if real_app == app:
                self._update_correct_guess(rank, app, is_comb)
                break
        else:
            if is_comb:
                self.comb_incorrect += 1
            else:
                self.incorrect += 1

        if is_comb and len(top_similarities) > 0:
            self.comb_len_of_candidates.append(len(top_similarities))
        elif len(top_similarities) > 0:
            self.len_of_candidates.append(len(top_similarities))

    def _update_correct_guess(self, guess_rank, app, is_comb=False):
        # Update stats based on which guess was correct.
        if is_comb:
            self.comb_correct[guess_rank - 1] += 1
        else:
            self.correct[guess_rank - 1] += 1

    def _log_no_similar_apps_found(self, real_app):
        with Logger() as logger:
            logger.warn(f"No similar apps found for {real_app}.")

    def display_statistics(self, is_comb=False, num_of_patterns=None):
        print("________________________________________________________")
        ja_version = self.ja_version
        print(
            f"Apriori with JA{ja_version} + JA{ja_version}S + SNI:"
            if is_comb
            else f"Apriori with JA{ja_version}:"
        )

        correct = sum(self.comb_correct) if is_comb else sum(self.correct)
        incorrect = self.comb_incorrect if is_comb else self.incorrect

        empty_candidates = (
            self.empty_comb_candidates if is_comb else self.empty_candidates
        )

        len_of_candidates = (
            self.comb_len_of_candidates if is_comb else self.len_of_candidates
        )

        pure_context = self.pure_context_comb if is_comb else self.pure_context

        empty_ja = self.empty_ja_comb if is_comb else self.empty_ja

        context_using_whole_db = (
            self.context_using_whole_db_comb if is_comb else self.context_using_whole_db
        )

        total = self.number_of_tls
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Empty context candidates: {empty_candidates}")
        print(f"Total: {total}\n")

        print(f"Accuracy overall: {round(correct / total, 4)}")
        print(f"Error rate: {round(incorrect / total, 4)}\n")

        correct_list = self.comb_correct if is_comb else self.correct
        accuracy_list = [round((count / total), 2) for count in correct_list]

        for i in range(self.candidate_size):
            print(f"{i + 1}. guess: {correct_list[i]} ({round(accuracy_list[i], 2)})")

        print(f"Empty JA candidates: {empty_ja} ({round(empty_ja / total, 2)})")
        print(f"Pure context: {pure_context} ({round(pure_context / total, 2)})")
        print(
            f"Context using whole db: {context_using_whole_db} ({round(context_using_whole_db / total, 2)})\n"
        )

        avg_len = sum(len_of_candidates) / len(len_of_candidates)
        median_len = np.median(len_of_candidates)
        modus_len = max(set(len_of_candidates), key=len_of_candidates.count)
        print(f"Average len of candidates: {round(avg_len, 4)}")
        print(f"Median len of candidates: {round(median_len, 4)}")
        print(f"Modus len of candidates: {round(modus_len, 4)}")
        print(f"Max len of candidates: {max(len_of_candidates)}")
        print(f"Min len of candidates: {min(len_of_candidates)}\n")

        if self.csv_file is not None:
            self.export_to_csv(
                [
                    [
                        is_comb,
                        config.columns_to_keep_for_context,
                        config.PATTERN_FILTERS,
                        self.min_support,
                        self.candidate_size,
                        ja_version,
                        correct,
                        incorrect,
                        empty_candidates,
                        num_of_patterns,
                        total,
                        correct / total,
                        incorrect / total,
                        avg_len,
                        median_len,
                        modus_len,
                        max(len_of_candidates),
                        min(len_of_candidates),
                        *correct_list,
                        *accuracy_list,
                    ]
                ],
                headers=[
                    "is_comb",
                    "items",
                    "pattern_filters",
                    "min_support",
                    "candidate_size",
                    "ja_version",
                    "correct",
                    "incorrect",
                    "empty_candidates",
                    "num_of_patterns",
                    "total",
                    "accuracy_overall",
                    "error_rate",
                    "avg_len_of_candidates",
                    "median_len_of_candidates",
                    "modus_len_of_candidates",
                    "max_len_of_candidates",
                    "min_len_of_candidates",
                    *[f"guess_{i + 1}" for i in range(len(correct_list))],
                    *[f"guess_perc_{i + 1}" for i in range(len(correct_list))],
                ],
            )

        if not is_comb:
            self.display_statistics(is_comb=True, num_of_patterns=num_of_patterns)

    def export_to_csv(self, data, headers=None):
        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode="a", newline="") as csvfile:
            csv_writer = csv.writer(
                csvfile,
                delimiter=";",
            )
            if headers and not file_exists:
                csv_writer.writerow(headers)
            for row in data:
                if isinstance(row, list):  # Ensure row is a list
                    csv_writer.writerow(row)
                else:
                    csv_writer.writerow([row])

    def identify(self, df):
        raise NotImplementedError("This method should be overridden by subclasses")


class Apriori(PatternMatchingMethod):
    def __str__(self):
        return

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
            tls_by_apps = data.groupby(config.APP_NAME)

            # Train for each app with multiple launches and look for frequent patterns over more launches
            for _, one_app_tls in tls_by_apps:
                self._train_group(one_app_tls, db)
            self.log_patterns(db)

    def log_patterns(self, db):
        with Logger() as logger:
            logger.debug("Frequent patterns found: \n")
            for app, patterns in db.frequent_patterns.items():
                logger.debug(f"app: {app}")
                logger.debug(f"patterns: {patterns}\n")

    def _init_db_for_app(self, app, db):
        if app in db.frequent_patterns:
            return
        else:
            db.frequent_patterns[app] = pd.DataFrame()
            with Logger() as logger:
                logger.debug(f"Creating new entry for {app}")

    def _add_patterns_to_db(self, app, patterns, db):
        """
        Add only UNIQUE frequent patterns to DB with normalized support.
        """
        # Remove duplicates
        patterns = patterns.drop_duplicates(subset="itemsets")

        # Sort by support
        patterns.sort_values(by="support", ascending=False, inplace=True)

        # map operator for dynamic configuration of filters
        ops = {
            "==": operator.eq,
            ">=": operator.ge,
            "<=": operator.le,
            "<": operator.lt,
            ">": operator.gt,
            "!=": operator.ne,
        }

        filtered_patterns = []

        # Extract filters from config
        for f in config.PATTERN_FILTERS:
            op = ops[f["operator"]]
            length = f["length"]
            head = f["head"]
            # Filter patterns based on length and operator
            subset = patterns[
                patterns["itemsets"].apply(lambda x: op(len(x), length))
            ].head(head)
            filtered_patterns.append(subset)

        if filtered_patterns:
            # Combine all subsets
            patterns = pd.concat(filtered_patterns, ignore_index=True)

        patterns = patterns.reset_index(drop=True)
        db.frequent_patterns[app] = pd.DataFrame(patterns)
        db.frequent_patterns[app] = self._normalize_support(db.frequent_patterns[app])
        with Logger() as logger:
            logger.debug(f"Found {len(patterns)} frequent item sets for {app} \n")

    def _normalize_support(self, patterns_df):
        patterns_df["normalized_support"] = np.log1p(
            patterns_df["support"]
        )  # Normalize support
        return patterns_df

    def _train_group(self, group, db):
        with Logger() as logger:
            app_name = group[config.APP_NAME].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            frequent_item_sets = self._execute_apriori(group)
            self._init_db_for_app(app_name, db)

            self._add_patterns_to_db(app_name, frequent_item_sets, db)

    def _preprocess(self, data):
        # Strip data of unnecessary columns.
        # Keep only columns that are needed for the Apriori algorithm.
        data = data.filter(config.columns_to_keep_for_context)
        data = data.astype(str)
        # Serialize the data.
        data_list = data.values.tolist()

        ###! The following code is based on [1] (see full citation at the top of the file) !###

        # Transform into one-hot encoding.
        te = TransactionEncoder()
        te_ary = te.fit(data_list).transform(data_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        ###! ---- END ---- !####
        # Convert to boolean values to ensure that the apriori algorithm works correctly,
        # as it requires the input data to be in a binary format.
        df_encoded = df.astype(bool)

        return df_encoded

    def _execute_apriori(self, group):
        processed_group = self._preprocess(group)
        with Logger() as logger:
            logger.info(
                f"Executing Apriori algorithm with min_support={self.min_support} ..."
            )
        freq_items_set = apriori(
            processed_group,
            min_support=self.min_support,
            use_colnames=True,
        )

        return freq_items_set

    def _jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def _overlap_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return intersection / len(set1) if len(set1) != 0 else 0

    def _dice_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        return (
            2 * intersection / (len(set1) + len(set2))
            if len(set1) + len(set2) != 0
            else 0
        )

    def _cosine_similarity(self, set1, set2):
        # Convert sets to bag-of-words representation
        all_items = list(set1.union(set2))
        vec1 = np.array([1 if item in set1 else 0 for item in all_items])
        vec2 = np.array([1 if item in set2 else 0 for item in all_items])

        # Compute Cosine Similarity
        return cosine_similarity([vec1], [vec2])[0][0]

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using Apriori algorithm ...")
            # Retrieve test data and group it by app.
            test_ds = db.get_test_df()
            test_ds_launches = test_ds.groupby(config.FILE)

            for _, launch in test_ds_launches:
                real_app = launch[config.APP_NAME].iloc[0]
                # Find similarity of tle entries in db of frequent patterns.
                top_guesses = self.find_similarity(db.frequent_patterns, launch)
                self._debug_identify_print(real_app, top_guesses)
                # Update statistics based on the results.
                self._update_statistics(real_app, top_guesses)

    def _debug_identify_print(self, real_app, top_guesses):
        if config.DEBUG_ENABLED:
            print(f"\033[1m{real_app}\033[0m:", end=" ")
            for app, similarity in top_guesses:
                if real_app == app:
                    print(f"\033[1;32m{app} {similarity:.2f}\033[0m", end="; ")
                else:
                    print(f"{app} {similarity:.2f}", end="; ")
            print()

    def _minmax_normalize(self, scores):
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())

        # Prevent division by zero (if all scores are the same)
        if min_score == max_score:
            return {k: 0.5 for k in scores}

        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

    def find_similarity(self, frequent_patterns, tls_group):
        if not frequent_patterns:
            return {}
        top_scores = {}
        pattern_df = defaultdict(int)

        stripped_tls = tls_group.filter(config.columns_to_keep_for_context)
        # print(stripped_tls)
        tls_set = frozenset(stripped_tls.values.flatten())
        total_apps = len(frequent_patterns)

        for app, patterns in frequent_patterns.items():
            for _, row in patterns.iterrows():
                pattern = frozenset(row["itemsets"])
                pattern_df[pattern] += 1

        for app, patterns in frequent_patterns.items():
            # Reset total score for each app
            total_score = 0

            for _, row in patterns.iterrows():
                pattern_set = frozenset(row["itemsets"])

                if not pattern_set:
                    continue

                df = pattern_df[pattern_set]
                idf = log(1 + total_apps / df)

                total_score += (
                    self._jaccard_similarity(pattern_set, tls_set) + 1
                ) * idf

                if pattern_set.issubset(tls_set):
                    total_score += (
                        len(pattern_set) * 10 * idf * (row["normalized_support"] + 1)
                    )

            if total_score > 0:
                top_scores[app] = total_score

        # Normalize scores using Min-Max Scaling
        norm_scores = self._minmax_normalize(top_scores)

        # Return top N apps with highest scores
        return heapq.nlargest(
            self.candidate_size, norm_scores.items(), key=lambda x: x[1]
        )

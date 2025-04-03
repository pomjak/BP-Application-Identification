"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 03/04/2025
"""

from database import Database
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from logger import Logger
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq

import constants as col_names


class PatternMatchingMethod:
    def __init__(self, min_sup, version):
        self.min_support = min_sup
        self.correct = 0
        self.incorrect = 0
        self.first_guess = 0
        self.second_guess = 0
        self.third_guess = 0

        self.comb_correct = 0
        self.comb_incorrect = 0
        self.comb_first_guess = 0
        self.comb_second_guess = 0
        self.comb_third_guess = 0

        self.empty_candidates = 0
        self.empty_comb_candidates = 0

        self.len_of_candidates = []
        self.comb_len_of_candidates = []

        self.number_of_tls = 0
        self.pure_context = 0
        self.pure_context_comb = 0

        self.empty_ja = 0
        self.empty_ja_comb = 0

        self.ja_version = version

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
        # If real app is 1st, 2nd or 3rd guess, update the statistics, else increment incorrect.
        for rank, (app, _) in enumerate(top_similarities[:3], start=1):
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
            attr_map = {
                1: "comb_first_guess",
                2: "comb_second_guess",
                3: "comb_third_guess",
            }
            correct_attr = "comb_correct"
        else:
            attr_map = {
                1: "first_guess",
                2: "second_guess",
                3: "third_guess",
            }
            correct_attr = "correct"

        if guess_rank in attr_map:
            setattr(self, attr_map[guess_rank], getattr(self, attr_map[guess_rank]) + 1)

        setattr(self, correct_attr, getattr(self, correct_attr) + 1)

    def _log_no_similar_apps_found(self, real_app):
        with Logger() as logger:
            logger.warn(f"No similar apps found for {real_app}.")

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

    def display_statistics(self, is_comb=False):
        print("________________________________________________________")
        ja_version = self.ja_version
        print(
            f"Apriori with JA{ja_version} + JA{ja_version}S + SNI:"
            if is_comb
            else f"Apriori with JA{ja_version}:"
        )

        correct = self.comb_correct if is_comb else self.correct
        incorrect = self.comb_incorrect if is_comb else self.incorrect

        empty_candidates = (
            self.empty_comb_candidates if is_comb else self.empty_candidates
        )

        first_guess = self.comb_first_guess if is_comb else self.first_guess
        second_guess = self.comb_second_guess if is_comb else self.second_guess
        third_guess = self.comb_third_guess if is_comb else self.third_guess

        len_of_candidates = (
            self.comb_len_of_candidates if is_comb else self.len_of_candidates
        )

        pure_context = self.pure_context_comb if is_comb else self.pure_context

        empty_ja = self.empty_ja_comb if is_comb else self.empty_ja

        total = self.number_of_tls
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Empty candidates: {empty_candidates}")
        print(f"Total: {total}\n")

        print(f"First guess: {first_guess} ({round(first_guess / correct, 2)})")
        print(f"Second guess: {second_guess} ({round(second_guess / correct, 2)})")
        print(f"Third guess: {third_guess} ({round(third_guess / correct, 2)})\n")

        print(f"Accuracy 1st guess : {round(first_guess / total, 4)}")
        print(f"Accuracy 2nd guess : {round(second_guess / total, 4)}")
        print(f"Accuracy 3rd guess : {round(third_guess / total, 4)}")
        print(f"Accuracy overall: {round(correct / total, 4)}")
        print(f"Error rate: {round(incorrect / total, 4)}\n")

        print(f"Empty JA candidates: {empty_ja} ({round(empty_ja / total, 2)})")
        print(f"Pure context: {pure_context} ({round(pure_context / total, 2)})\n")

        avg_len = sum(len_of_candidates) / len(len_of_candidates)
        median_len = np.median(len_of_candidates)
        modus_len = max(set(len_of_candidates), key=len_of_candidates.count)
        print(f"Average len of candidates: {avg_len}")
        print(f"Median len of candidates: {median_len}")
        print(f"Modus len of candidates: {modus_len}")
        print(f"Max len of candidates: {max(len_of_candidates)}")
        print(f"Min len of candidates: {min(len_of_candidates)}\n")

        if not is_comb:
            self.display_statistics(is_comb=True)

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
            self.log_patterns(db)
            # exit(0)

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
        Add only UNIQUE frequent patterns to DB with support normalized to percentile rank.
        """
        patterns = patterns.drop_duplicates(subset="itemsets")  # Remove duplicates

        # median = patterns["support"].median()
        # patterns = patterns.drop(patterns[patterns["support"] < median].index)
        # patterns = patterns.reset_index(drop=True)

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
            test_ds_launches = test_ds.groupby(col_names.FILE)

            for _, launch in test_ds_launches:
                real_app = launch[col_names.APP_NAME].iloc[0]
                # Find similarity of tle entries in db of frequent patterns.
                top_guesses = self.find_similarity(db.frequent_patterns, launch)
                self._debug_identify_print(real_app, top_guesses)
                # Update statistics based on the results.
                self._update_statistics(real_app, top_guesses)

    def _debug_identify_print(self, real_app, top_guesses, warn=False):
        if col_names.DEBUG_ENABLED:
            print(f"\033[1m{real_app}\033[0m:", end=" ")
            if warn:
                for app, similarity in top_guesses:
                    if real_app == app:
                        print(f"\033[1;32m{app} {similarity:.2f}\033[0m", end="; ")
                    else:
                        print(f"\033[1;33m{app} {similarity:.2f}\033[0m", end="; ")
                print()
            else:
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
        top_scores = {}

        stripped_tls = tls_group.drop(columns=[col_names.FILE, col_names.APP_NAME])
        tls_set = frozenset(stripped_tls.values.flatten())

        pattern_counts = {
            app: len(patterns) for app, patterns in frequent_patterns.items()
        }

        for app, patterns in frequent_patterns.items():
            # Reset total score for each app
            total_score = 0
            num_patterns = pattern_counts[app]

            for _, row in patterns.iterrows():
                pattern_set = frozenset(row["itemsets"])

                jaccard = self._jaccard_similarity(tls_set, pattern_set)
                overlap = self._overlap_similarity(tls_set, pattern_set)
                dice = self._dice_similarity(tls_set, pattern_set)

                bonus_score = 50 if pattern_set.issubset(tls_set) else 1

                combined_score = jaccard * 0.3 + overlap * 0.5 + dice * 0.2

                total_score += (
                    combined_score * (row["normalized_support"] + 1) * bonus_score
                )
            # Adjust score based on the number of patterns
            adjusted_score = total_score / ((np.log1p(num_patterns) + 1) ** 2.0)

            if adjusted_score > 0:
                top_scores[app] = adjusted_score

        # Normalize scores using Min-Max Scaling
        norm_scores = self._minmax_normalize(top_scores)

        # Return top 3 apps with highest scores
        return heapq.nlargest(3, norm_scores.items(), key=lambda x: x[1])

"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 11/03/2025
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
        self.first_guess = 0
        self.second_guess = 0
        self.third_guess = 0
        self.incorrect = 0
        self.uniq = 0

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
        print()
        print(f"Error rate: {round(self.incorrect / (total), 4)}")
        print(f"Number of unique patterns sets: {self.uniq}")

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
                logger.debug(f"patterns: {db.frequent_patterns[app]}\n")

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

        return apriori(processed_group, min_support=0.01, use_colnames=True, max_len=3)

    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

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

                # sort similarities in descending order and get the top 3
                top_similarities = sorted(similarities, reverse=True)[:3]
                real_app = group[col_names.APP_NAME].iloc[0]

                logger.info(
                    f" real app: {real_app}, top 3 similarities: {[(round(sim, 2), app, index) for sim, app, index in top_similarities]}"
                )

                self._update_statistics(real_app, top_similarities)

            self.uniq = self.get_number_of_unique_patterns_sets(db.frequent_patterns)
            for app in db.frequent_patterns:
                print(
                    f"app: {app}, unique patterns: {self.get_number_of_unique_patterns_sets(db.frequent_patterns, app)}"
                )

    def _update_statistics(self, real_app, top_similarities):
        # check top 3 guesses
        if top_similarities:
            if real_app == top_similarities[0][1]:
                self.first_guess += 1
                self.correct += 1
            elif len(top_similarities) > 1 and real_app == top_similarities[1][1]:
                self.second_guess += 1
                self.correct += 1
            elif len(top_similarities) > 2 and real_app == top_similarities[2][1]:
                self.third_guess += 1
                self.correct += 1
            else:
                self.incorrect += 1
        else:
            with Logger() as logger:
                logger.warn("No similar apps found")

    """
    Returns the number of unique patterns sets in the database if app is None, 
    otherwise returns the number of unique patterns sets for the given app.
    """

    def get_number_of_unique_patterns_sets(self, trained_patterns, app=None):
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

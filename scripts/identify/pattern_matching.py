"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 05/03/2025
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

    def display_statistics(self):
        print("________________________________________________________")
        print("Pattern matching:")
        print(f"Correct: {self.correct}")
        print(f"Incorrect: {self.incorrect}")
        print(f"Accuracy: {self.correct / (self.correct + self.incorrect)}")
        print(f"Error rate: {self.incorrect / (self.correct + self.incorrect)}")
        print(f"Total: {self.correct + self.incorrect}")
        pass

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

            self._log_patterns(db)

    def _log_patterns(self, db):
        with Logger() as logger:
            logger.debug("Frequent patterns found: \n")
            for app in db.frequent_patterns:
                logger.debug(f"app: {app}\n")
                logger.debug(f"patterns: {db.frequent_patterns[app]}\n")

    def _train_group(self, group, db):
        with Logger() as logger:
            app_name = group[col_names.APP_NAME].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            processed = self._preprocess(group)
            frequent_item_sets = apriori(processed, min_support=0.5, use_colnames=True)

            if app_name not in db.frequent_patterns:
                db.frequent_patterns[app_name] = pd.DataFrame()
                logger.debug(f"Creating new entry for {app_name}")

            db.frequent_patterns[app_name] = frequent_item_sets
            logger.debug(
                f"Found {len(frequent_item_sets)} frequent item sets for {app_name} \n"
            )

    def _preprocess(self, data):
        data = data.drop(columns=[col_names.FILE, col_names.APP_NAME])
        data = data.astype(str)

        # serialize the data
        data_list = data.values.tolist()

        # transform into one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(data_list).transform(data_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # convert to boolean values to ensure that the apriori algorithm works correctly,
        # as it requires the input data to be in a binary format
        df_encoded = df.astype(bool)

        return df_encoded

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using Apriori algorithm ...")

            test_ds = db.get_test_df()
            groups = test_ds.groupby(col_names.FILE)
            for _, group in groups:
                if len(group) > 1:
                    processed_group = self._preprocess(group)

                    # iterate over the test dataset
                    found_item_sets = apriori(
                        processed_group, min_support=0.5, use_colnames=True
                    )

                    # check if the found frequent item sets are in the training dataset
                    found_item_set = set(found_item_sets["itemsets"])
                    max_similarity = 0
                    for app in db.frequent_patterns:
                        trained_items_set = set(db.frequent_patterns[app]["itemsets"])
                        if len(found_item_set):
                            similarity = len(
                                found_item_set.intersection(trained_items_set)
                            ) / len(found_item_set.union(trained_items_set))
                            if similarity > max_similarity:
                                max_similarity = similarity
                                max_app = app

                    logger.info(
                        f"max similarity: {round(max_similarity, 2)}, max app: {max_app}, real app: {group[col_names.APP_NAME].iloc[0]}\n"
                    )

                    if max_app == group[col_names.APP_NAME].iloc[0]:
                        self.correct += 1
                    else:
                        self.incorrect += 1


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

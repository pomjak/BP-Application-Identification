"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 03/03/2025
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

            for app in db.frequent_patterns:
                logger.debug(f"app: {app}\n")
                for freq_item_set in db.frequent_patterns[app]:
                    logger.debug(f"patterns: {freq_item_set}\n")

    def _train_group(self, group, db):
        with Logger() as logger:
            group = group.astype(str)

            app_name = group[col_names.APP_NAME].iloc[0]
            logger.debug(f"Training for {app_name}, with length of {len(group)}")

            processed = self.preprocess(group)

            frequent_item_sets = apriori(processed, min_support=0.5, use_colnames=True)

            if app_name not in db.frequent_patterns:
                db.frequent_patterns[app_name] = []
                logger.debug(f"Creating new entry for {app_name}")

            db.frequent_patterns[app_name].append(frequent_item_sets)
            logger.debug(
                f"Found {len(frequent_item_sets)} frequent item sets for {app_name} \n"
            )

    def preprocess(self, data):
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

    def identify(self, db):
        pass


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

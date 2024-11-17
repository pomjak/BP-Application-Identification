"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/11/2024
"""

import constants as col_names
from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


class Database:
    def __init__(self, dataset):
        with Logger() as logger:
            self.dataset = dataset
            logger.info("Parsing dataset ...")
            self.df = pd.read_csv(self.dataset, delimiter=";")

            # Split the dataset into training and testing data frames
            self.train_df, self.test_df = train_test_split(self.df)
            logger.info(
                f"Split dataset {len(self.train_df)} training and {len(self.test_df)} testing samples"
            )

            self.lookup_table = {}

    def get_dataframe(self):
        return self.df

    def create_lookup_table(self, ja_version):
        with Logger() as logger:
            logger.info("Creating lookup table ...")
            self.lookup_table = {}

            for _, row in self.train_df.iterrows():
                app_name = row[col_names.APP_NAME]
                # lookup table for JA3/4 fingerprints and corresponding application names as set of strings

                if ja_version == 4:
                    ja4hash = row[col_names.JA4]

                    if ja4hash in self.lookup_table:
                        self.lookup_table[ja4hash].add(app_name)
                    else:
                        self.lookup_table[ja4hash] = {app_name}
                        
                else:
                    ja3hash = row[col_names.JA3]

                    if ja3hash in self.lookup_table:
                        self.lookup_table[ja3hash].add(app_name)
                    else:
                        self.lookup_table[ja3hash] = {app_name}

    def get_app(self, ja_hash):
        return self.lookup_table.get(ja_hash, set())

"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 18/11/2024
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
            if ja_version == 4:
                self.lookup_table[col_names.JA4] = {}
                self.lookup_table[col_names.JA4_S] = {}
                self.lookup_table[col_names.SNI] = {}
            else:
                self.lookup_table[col_names.JA3] = {}
                self.lookup_table[col_names.JA3_S] = {}
                self.lookup_table[col_names.SNI] = {}

            for _, row in self.train_df.iterrows():
                app_name = row[col_names.APP_NAME]
                # lookup table for JA3/4 fingerprints and corresponding application names as set of strings

                sni = row[col_names.SNI]
                if sni in self.lookup_table[col_names.SNI]:
                    self.lookup_table[col_names.SNI][sni].add(app_name)
                else:
                    self.lookup_table[col_names.SNI][sni] = {app_name}
                        
                if ja_version == 4:
                    ja4hash = row[col_names.JA4]
                    ja4s = row[col_names.JA4_S]

                    if ja4hash in self.lookup_table[col_names.JA4]:
                        self.lookup_table[col_names.JA4][ja4hash].add(app_name)
                    else:
                        self.lookup_table[col_names.JA4][ja4hash] = {app_name}

                    if ja4s in self.lookup_table[col_names.JA4_S]:
                        self.lookup_table[col_names.JA4_S][ja4s].add(app_name)
                    else:
                        self.lookup_table[col_names.JA4_S][ja4s] = {app_name}

                else:
                    ja3hash = row[col_names.JA3]
                    ja3s = row[col_names.JA3_S]

                    if ja3hash in self.lookup_table[col_names.JA3]:
                        self.lookup_table[col_names.JA3][ja3hash].add(app_name)
                    else:
                        self.lookup_table[col_names.JA3][ja3hash] = {app_name}

                    if ja3s in self.lookup_table[col_names.JA3_S]:
                        self.lookup_table[col_names.JA3_S][ja3s].add(app_name)
                    else:
                        self.lookup_table[col_names.JA3_S][ja3s] = {app_name}

    def log_lookup_table(self):
        with Logger() as logger:
            for table in self.lookup_table.values():
                logger.info("Printing lookup table ...")
                for key, value in table.items():
                    logger.debug(f"Key: {key}, Value: {value}")
            logger.debug(f"{len(self.lookup_table)} keys in lookup table")

    def get_app(self, type, value):
        return self.lookup_table[type].get(value, set())

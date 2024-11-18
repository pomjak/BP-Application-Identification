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
            self.lookup_table = {}
            self.train_df = None
            self.test_df = None
            self.ja_version = None

            logger.info("Parsing dataset ...")
            try:
                df = pd.read_csv(self.dataset, delimiter=";")
            except FileNotFoundError:
                logger.error("File not found.")
                print("File not found.")
                exit(1)

            # Split the dataset into training and testing data frames
            self.train_df, self.test_df = train_test_split(df, train_size=0.8)
            logger.info(f"training dataset: {len(self.train_df)}")
            logger.info(f"testing dataset: {len(self.test_df)}")

    def create_lookup_table(self, ja_version):
        with Logger() as logger:
            logger.info("Creating lookup table ...")

            self.ja_version = ja_version
            ja_keys = col_names.get_keys(ja_version)

            # init lookup tables for every
            self.lookup_table = {key: {} for key in ja_keys}

            for index, row in self.train_df.iterrows():
                # get app name and ja keys
                app_name = row[col_names.APP_NAME]

                # insert ja, jas fingerprints and sni 
                for key in ja_keys:
                    # if item from csv is missing, ignore
                    if pd.notna(row[key]):
                        self.__update_lookup_table(key, row[key], app_name)

    def __update_lookup_table(self, key, value, app_name):
        # if exists update the set, else create a new set
        if value in self.lookup_table[key]:
            self.lookup_table[key][value].add(app_name)
        else:
            self.lookup_table[key][value] = {app_name}

    def log_lookup_table(self):
        with Logger() as logger:
            logger.debug("Printing lookup tables ...")
            for col in col_names.get_keys(self.ja_version):
                logger.info(f"Table: {col}")
                for key, value in self.lookup_table[col].items():
                    logger.debug(f"key: {key}, value: {value}")

    def get_app(self, type, value):
        return self.lookup_table[type].get(value, set())

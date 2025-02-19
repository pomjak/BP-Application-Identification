"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 19/02/2025
"""

import constants as col_names
from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


class Database:
    def __init__(self, dataset):
        with Logger() as logger:
            self.dataset = dataset
            self.lookup_table = {}  # lookup table for fingerprinting
            self.frequent_patterns = {}  # lookup table for frequent patterns
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

            df = df.drop(
                df[df[col_names.TYPE] == "A"].index
            )  # filter out rows with type A
            self.split_dataset(df)
            logger.info(f"training dataset: {len(self.train_df)}")
            logger.info(f"testing dataset: {len(self.test_df)}")

    def split_dataset(self, df):
        train_list = []
        test_list = []

        grouped = df.groupby(col_names.FILE)

        for _, group in grouped:
            if len(group) > 1:
                train_group, test_group = train_test_split(
                    group, train_size=0.8, shuffle=False
                )
                train_list.append(train_group)
                test_list.append(test_group)

        self.train_df = pd.concat(train_list)
        self.test_df = pd.concat(test_list)

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

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

#!/usr/bin/python3
"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 01/03/2025
"""

import constants as col_names
from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


class Database:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None
        self.lookup_table = {}  # lookup table for fingerprinting
        self.frequent_patterns = {}  # lookup table for frequent patterns
        self.train_df = []
        self.test_df = None
        self.ja_version = None

        self.handle_file(dataset)
        self.split_dataset()

    def handle_file(self, file):
        with Logger() as logger:
            logger.info("Parsing dataset ...")
            try:
                self.df = pd.read_csv(file, delimiter=";")
            except FileNotFoundError:
                logger.error("File not found.")
                print("File not found.")
                exit(1)
            except pd.errors.EmptyDataError:
                logger.error("File is empty.")
                print("File is empty.")
                exit(1)

    def split_dataset(self):
        with Logger() as logger:
            # filter out rows with type A
            self.df.drop(self.df[self.df[col_names.TYPE] == "A"].index, inplace=True)
            logger.info("TYPE A rows filtered out.")

            groups = self.df.groupby(col_names.FILE)
            for _, group in groups:
                logger.debug(f"Group: {group}")
                if len(group) > 1:
                    self.train_df, self.test_df = train_test_split(
                        group, test_size=0.2, shuffle=False
                    )

            logger.info(f"training dataset: {len(self.train_df)}")
            logger.info(f"{self.train_df}")
            logger.info(f"testing dataset: {len(self.test_df)}")
            logger.info(f"{self.test_df}")

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

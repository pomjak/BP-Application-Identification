#!/usr/bin/python3
"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 06/04/2025
"""

import constants as col_names
from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


class Database:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = {}
        self.lookup_table = {}  # lookup table for fingerprinting
        self.frequent_patterns = {}  # lookup table for frequent patterns
        self.train_df = {}
        self.test_df = {}
        self.ja_version = None
        self.context_results = {}
        self.fingerprinting_results = {}

        self.handle_file(dataset)
        self.filter_out_dataset()
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

    def filter_out_dataset(self):
        with Logger() as logger:
            # filter out rows with type A
            self.df.drop(self.df[self.df[col_names.TYPE] == "A"].index, inplace=True)
            logger.info("TYPE A rows filtered out.")

            # drop everything except these columns
            self.df = self.df.filter(
                [
                    col_names.APP_NAME,
                    col_names.FILE,
                    col_names.JA3,
                    col_names.JA3_S,
                    col_names.JA4,
                    col_names.JA4_S,
                    col_names.SNI,
                    col_names.ORG,
                    # col_names.TLS_VERSION,
                    # col_names.CIPHER_SUITE,
                    # col_names.CLIENT_EXT,
                    # col_names.CLIENT_SUPPORTED_GROUPS,
                    # col_names.CLIENT_SUPPORTED_VERSIONS,
                    # col_names.EC_FMT,
                    # col_names.ALPN,
                    # col_names.SIGNATURE_ALGORITHMS,
                    # col_names.SERVER_CIPHER_SUITE,
                    # col_names.SERVER_EXTENSIONS,
                    # col_names.SERVER_SUPPORTED_VERSIONS,
                ]
            )

    def split_dataset(self):
        with Logger() as logger:
            train_list = []
            test_list = []

            groups = self.df.groupby(col_names.FILE)
            single_occurrence = 0
            for _, group in groups:
                if len(group) > 1:
                    train_group, test_group = train_test_split(
                        group, test_size=0.25, shuffle=False
                    )
                    train_list.append(train_group)
                    test_list.append(test_group)

                else:
                    single_occurrence += 1
                    logger.warn(
                        f"File: {group[col_names.FILE].values[0]} has only one row. Occurrence: {single_occurrence}"
                    )
                    train_list.append(group)

            self.train_df = pd.concat(train_list)
            self.train_df.drop_duplicates(inplace=True)
            self.test_df = pd.concat(test_list)

            logger.info(f"training dataset: {len(self.train_df)}")
            logger.debug(f"{self.train_df}")

            logger.info(f"testing dataset: {len(self.test_df)}")
            logger.debug(f"{self.test_df}")

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

"""
File: database.py
Description: This file contains databases for storing fingerprints.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 16/11/2024
"""

from logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


class Database:
    def __init__(self, dataset):
        self.dataset = dataset
        with Logger() as logger:
            logger.info("Parsing dataset ...")
            self.df = pd.read_csv(self.dataset, delimiter=";")

            # Split the dataset into training and testing data frames
            self.train_df, self.test_df = train_test_split(self.df)
            logger.info(
                f"Split dataset {len(self.train_df)} training and {len(self.test_df)} testing samples"
            )

    def get_dataframe(self):
        return self.df

    def get_app(self, col, value):
        return set(self.train_df[self.train_df[col] == value]["AppName"].tolist())

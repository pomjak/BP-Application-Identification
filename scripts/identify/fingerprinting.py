"""
File: config.py
Description: This file contains methods for identification of applications using JA3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/11/2024
"""

import constants as col_names
from database import Database


class FingerprintingMethod:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def statistics(self):
        return self.correct, self.incorrect, self.correct + self.incorrect

    def display_statistics(self):
        correct, incorrect, total = self.statistics()
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct/total}")

    def identify(self, db: Database):
        raise NotImplementedError("This method should be overridden by derived classes")


class JA3(FingerprintingMethod):
    def identify(self, db: Database):
        # iterate over test dataset and check if app name is in set of candidates
        for _, row in db.test_df.iterrows():
            # extract JA3 hash and app name from one row of ds
            ja3hash = row[col_names.JA3]
            appname = row[col_names.APP_NAME]

            if appname in db.get_app(ja3hash):
                self.correct += 1
            else:
                self.incorrect += 1


class JA4(FingerprintingMethod):
    def identify(self, db: Database):
        for _, row in db.test_df.iterrows():
            ja4hash = row[col_names.JA4]
            appname = row[col_names.APP_NAME]

            if appname in db.get_app(ja4hash):
                self.correct += 1
            else:
                self.incorrect += 1

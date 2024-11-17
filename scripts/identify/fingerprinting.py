"""
File: config.py
Description: This file contains methods for identification of applications using JA3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/11/2024
"""
from database import Database


class FingerprintingMethod:
    JA3_COL_NAME = "JA3hash"
    JA3S_COL_NAME = "JA3Shash"
    JA4_COL_NAME = "JA4hash"
    JA4S_COL_NAME = "JA4Shash"
    APP_NAME_COL_NAME = "AppName"

    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def statistics(self):
        return self.correct, self.incorrect, self.correct + self.incorrect

    def display_statistics(self):
        correct, incorrect, total = self.statistics()
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")

    def fingerprint(self, db: Database):
        raise NotImplementedError("This method should be overridden by derived classes")


class JA3(FingerprintingMethod):
    def identify(self, db: Database):
        for row in db.test_df.iterrows():
            ja3hash = row[1][self.JA3_COL_NAME]
            appname = row[1][self.APP_NAME_COL_NAME]

            if appname in db.get_app(self.JA3_COL_NAME, ja3hash):
                self.correct += 1
            else:
                self.incorrect += 1


class JA4(FingerprintingMethod):
    def identify(self, db: Database):
        for row in db.test_df.iterrows():
            ja4hash = row[1][self.JA4_COL_NAME]
            appname = row[1][self.APP_NAME_COL_NAME]

            if appname in db.get_app(self.JA4_COL_NAME, ja4hash):
                self.correct += 1
            else:
                self.incorrect += 1

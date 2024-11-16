"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 16/11/2024
"""

from logger import Logger
from config import config
from database import Database
from fingerprinting import JA3, JA4
from pattern_matching import Apriori, SPADE, PrefixSpan


def main():
    with Logger() as logger:
        logger.info("Starting identification process...")

        db = Database(config.dataset)
        ja3_correct = 0
        ja3_incorrect = 0
        ja4_correct = 0
        ja4_incorrect = 0

        for row in db.test_df.iterrows():
            ja3hash = row[1]["JA3hash"]
            appname = row[1]["AppName"]
            ja4hash = row[1]["JA4hash"]

            if appname in db.get_app("JA3hash", ja3hash):
                ja3_correct += 1
            else:
                ja3_incorrect += 1

            if appname in db.get_app("JA4hash", ja4hash):
                ja4_correct += 1
            else:
                ja4_incorrect += 1

        print(
            f"JA3: Correct: {ja3_correct}, Incorrect: {ja3_incorrect}, Total: {ja3_correct + ja3_incorrect}"
        )
        print(
            f"JA4: Correct: {ja4_correct}, Incorrect: {ja4_incorrect}, Total: {ja4_correct + ja4_incorrect}"
        )
        print(f"JA3: Accuracy: {ja3_correct / (ja3_correct + ja3_incorrect) * 100}%")
        print(f"JA4: Accuracy: {ja4_correct / (ja4_correct + ja4_incorrect) * 100}%")


if __name__ == "__main__":
    main()

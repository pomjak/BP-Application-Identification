"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/11/2024
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

        logger.info("Selecting JA version ...")
        if config.ja_version == 4:
            fingerprinting = JA4()
        else:
            fingerprinting = JA3()

        logger.info("Identifying using fingerprinting method...")
        fingerprinting.identify(db)
        fingerprinting.display_statistics()


if __name__ == "__main__":
    main()

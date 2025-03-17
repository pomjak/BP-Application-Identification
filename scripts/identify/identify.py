"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/03/2025
"""

from logger import Logger
from config import Config
from database import Database
from fingerprinting import FingerprintingMethod
from pattern_matching import Apriori
import time
from result_merger import ResultMerger


def main():
    with Logger() as logger:
        logger.info("[START]")

        config = Config()
        db = Database(config.dataset)

        logger.info(f"Selecting JA{config.ja_version} version")

        fingerprinting = FingerprintingMethod(config.ja_version)
        db.create_lookup_table(config.ja_version)

        logger.info("Identifying using fingerprinting method...")

        fingerprinting.identify(db)
        fingerprinting.display_statistics()

        context = Apriori()

        context.train(db)
        context.identify(db)
        context.display_statistics()

        result_merger = ResultMerger()
        result_merger.merge(db.fingerprinting_results, db.context_results, db.test_df)
        result_merger.display_statistics()

        logger.info("[FINISH]")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))

"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 01/03/2025
"""

from logger import Logger
from config import Config
from database import Database
from fingerprinting import JA3, JA4
from pattern_matching import Apriori, SPADE, PrefixSpan
import time


def main():
    with Logger() as logger:
        logger.info("[START]")

        config = Config()
        db = Database(config.dataset)

        logger.info("Selecting JA version ...")

        if config.ja_version == 4:
            fingerprinting = JA4()
            db.create_lookup_table(4)
        else:
            fingerprinting = JA3()
            db.create_lookup_table(3)

        logger.info("Identifying using fingerprinting method...")

        fingerprinting.identify(db)
        fingerprinting.display_statistics()
        # db.log_lookup_table()  # [DEBUG]

        # match config.pattern_algorithm:
        #     case "apriori":
        #         context = Apriori()
        #     case "prefixspan":
        #         context = PrefixSpan()
        #     case "spade":
        #         context = SPADE()
        context = Apriori()

        logger.info("Identifying using pattern searching algos...")
        context.train(db)
        context.identify(db)
        context.display_statistics()

        logger.info("[FINISH]")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))

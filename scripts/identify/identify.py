"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 03/04/2025
"""

from logger import Logger
from config import Config
from database import Database
from fingerprinting import FingerprintingMethod
from pattern_matching import Apriori
from ja_context import JA_Context
import time


def main():
    with Logger() as logger:
        logger.info("[START]")

        config = Config()
        db = Database(config.dataset)

        fingerprinting = FingerprintingMethod(config.ja_version)
        db.create_lookup_table(config.ja_version)
        fingerprinting.identify(db)
        fingerprinting.display_statistics()

        context = Apriori(
            config.min_support,
            config.ja_version,
            config.max_candidates_length,
        )
        context.train(db)

        ja_context = JA_Context(
            fingerprinting,
            context,
            config.sliding_window_size,
            config.max_candidates_length,
        )
        ja_context.identify(db)
        ja_context.display_statistics()
        logger.info("[FINISH]")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))

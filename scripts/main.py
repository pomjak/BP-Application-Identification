"""
File: main.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 11/05/2025
"""

from identify.command_line_parser import CommandLineParser
from identify.logger import Logger
from identify.database import Database
from identify.fingerprinting import FingerprintingMethod
from identify.pattern_matching import Apriori
from identify.ja_context import JA_Context
import time


def main():
    with Logger() as logger:
        logger.info("[START]")

        config = CommandLineParser()
        db = Database(config.dataset)

        fingerprinting = FingerprintingMethod(config.ja_version)
        db.create_lookup_table(config.ja_version)
        fingerprinting.identify(db)
        fingerprinting.display_statistics()

        context = Apriori(
            config.min_support,
            config.ja_version,
            config.max_candidates_length,
            csv_file=config.csv_file,
        )
        context.train(db)

        ja_context = JA_Context(
            fingerprinting,
            context,
            config.sliding_window_size,
        )
        start_time = time.time()
        ja_context.identify(db)
        finish_time = time.time() - start_time
        ja_context.context.display_statistics(
            time=finish_time, win=config.sliding_window_size
        )
        print("--- identification took %s seconds ---" % round(finish_time, 2))
        logger.info("[FINISH]")


if __name__ == "__main__":
    main()

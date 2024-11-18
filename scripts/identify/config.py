"""
File: config.py
Description: This file contains the config class which parses and stores command-line arguments.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 18/11/2024
"""

import argparse
from logger import Logger


class Config:
    def __init__(self):
        with Logger() as logger:
            logger.info("Parsing command-line arguments...")

            args = self.parse_arguments()
            self.dataset = args.dataset

            logger.info(f"Dataset path set: {self.dataset}")

            self.ja_version = args.ja_version
            logger.info(f"JA version set: {self.ja_version}")

            self.pattern_algorithm = args.pattern_algorithm
            self.map_algos()
            logger.info(f"Pattern algorithm set: {self.pattern_algorithm}")

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description="Identify applications using JA3/4 fingerprints and frequent pattern matching algorithms in network traffic"
        )
        parser.add_argument(
            "-d", "--dataset", type=str, required=True, help="Path to the dataset"
        )

        parser.add_argument(
            "-f",
            "--ja_version",
            type=int,
            help="version of fingerprinting [JA3 or JA4]",
            choices=[3, 4],
            default=4,
        )

        parser.add_argument(
            "-p",
            "--pattern_algorithm",
            type=str,
            help="type of pattern searching algorithm [apriori, prefixspan or spade]",
            choices=["apriori", "a", "prefixspan", "p", "spade", "s"],
            default="prefixspan",
        )

        return parser.parse_args()

    # map abbreviations to complete names
    def map_algos(self):
        pattern_map = {"a": "apriori", "p": "prefixspan", "s": "spade"}
        if self.pattern_algorithm in pattern_map:
            self.pattern_algorithm = pattern_map[self.pattern_algorithm]

"""
File: config.py
Description: This file contains the Config class which parses and stores command-line arguments.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
"""

import argparse


class Config:
    def __init__(self):
        args = self.parse_arguments()
        self.ds = args.dataset
        self.ja = args.ja_version
        self.pattern_algo = args.pattern_algorithm
        self.map_algos()

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

    def map_algos(self):
        pattern_map = {"a": "apriori", "p": "prefixspan", "s": "spade"}
        if self.pattern_algo in pattern_map:
            self.pattern_algo = pattern_map[self.pattern_algo]


config = Config()

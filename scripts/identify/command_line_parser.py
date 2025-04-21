"""
File: command_line_parser.py
Description: This file contains the config class which parses and stores command-line arguments.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 04/04/2025
"""

import argparse
from .logger import Logger


class CommandLineParser:
    def __init__(self):
        with Logger() as logger:
            logger.info("Parsing command-line arguments...")

            args = self.__parse_arguments()
            self.dataset = args.dataset

            logger.info(f"Dataset path set: {self.dataset}")

            self.ja_version = args.ja_version
            logger.info(f"JA version set: {self.ja_version}")

            self.sliding_window_size = args.sliding_window_size
            logger.info(f"Sliding window size set: {self.sliding_window_size}")

            self.min_support = args.min_support
            logger.info(f"Minimum support set: {self.min_support}")

            self.max_candidates_length = args.max_candidates_length
            logger.info(f"Maximum candidates length set: {self.max_candidates_length}")

    def __parse_arguments(self):
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
            "-w",
            "--sliding_window_size",
            type=int,
            help="size of sliding window",
            default=10,
        )

        parser.add_argument(
            "-m",
            "--min_support",
            type=float,
            help="minimum support for frequent pattern mining",
            default=0.1,
        )

        parser.add_argument(
            "-c",
            "--max_candidates_length",
            type=int,
            help="maximum length of candidate patterns",
            default=4,
        )

        return parser.parse_args()

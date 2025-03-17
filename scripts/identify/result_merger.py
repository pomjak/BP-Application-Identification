"""
File: result_merger.py
Description: This file contains methods for merging results from fingerprinting and frequent pattern matching.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 17/03/2025
Updated: 17/03/2025
"""

from logger import Logger
from constants import APP_NAME


class ResultMerger:
    def __init__(self):
        pass

    def display_statistics(self):
        pass

    def merge(self, fingerprinting_results, context_results, test_df):
        with Logger() as logger:
            if len(fingerprinting_results) != len(context_results):
                logger.error("Results are not the same length.")

            for index, row in test_df.iterrows():
                logger.debug(f"App: {row[APP_NAME]}")
                ja_candidates = fingerprinting_results[index]["ja_candidates"]
                combined_candidates = fingerprinting_results[index][
                    "combined_candidates"
                ]
                logger.debug(f"candidate: {ja_candidates}")
                logger.debug(f"combined_candidates: {combined_candidates}")
                logger.debug(f"apriori: {context_results[index]}")

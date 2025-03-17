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
        self.correct = 0
        self.incorrect = 0
        self.correct_len = 0
        self.incorrect_len = 0

    def display_statistics(self):
        total = self.correct + self.incorrect
        print("________________________________________________________")
        print("Fingerprinting with context:")
        print(f"Correct: {self.correct}")
        print(f"Incorrect: {self.incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy overall: {round(self.correct / (total), 4)}")
        print(f"Error rate: {round(self.incorrect / (total), 4)}")
        print(
            f"Average number of candidates: {round((self.correct_len + self.incorrect_len) / total, 2)}"
        )

    def merge(self, fingerprinting_results, context_results, test_df):
        with Logger() as logger:
            if len(fingerprinting_results) != len(context_results):
                logger.error("Results are not the same length.")

            for index, row in test_df.iterrows():
                # Get candidates from fingerprinting and context matching
                fingerprinting_candidates = fingerprinting_results[index][
                    "combined_candidates"
                ]
                # Get candidates from context matching.
                context_candidates = context_results[index]

                logger.debug(
                    f"{row[APP_NAME]}:{context_candidates}:{fingerprinting_candidates}"
                )
                final_candidates = fingerprinting_candidates.union(context_candidates)
                if row[APP_NAME] in final_candidates:
                    self.correct += 1
                    self.correct_len += len(final_candidates)
                else:
                    self.incorrect += 1
                    self.incorrect_len += len(final_candidates)

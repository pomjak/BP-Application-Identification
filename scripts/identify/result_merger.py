"""
File: result_merger.py
Description: This file contains methods for merging results from fingerprinting and frequent pattern matching.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 17/03/2025
Updated: 18/03/2025
"""

from logger import Logger
from constants import APP_NAME


class ResultMerger:
    def __init__(self, version):
        self.version = version
        self.correct = 0
        self.incorrect = 0
        self.can_len = 0

        self.correct_comb = 0
        self.incorrect_comb = 0
        self.can_comb_len = 0

    def display_statistics(self):
        total = self.correct + self.incorrect
        print("________________________________________________________")
        print(f"JA{self.version}+context:")
        print(f"Correct: {self.correct}")
        print(f"Incorrect: {self.incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy overall: {round(self.correct / (total), 4)}")
        print(f"Error rate: {round(self.incorrect / (total), 4)}")
        print(f"Average number of candidates: {round(self.can_len / total, 2)}")

        total = self.correct_comb + self.incorrect_comb
        print("________________________________________________________")
        print(f"JA{self.version}+JA{self.version}+SNI+context:")
        print(f"Correct: {self.correct_comb}")
        print(f"Incorrect: {self.incorrect_comb}")
        print(f"Total: {total}")
        print(f"Accuracy overall: {round(self.correct_comb / (total), 4)}")
        print(f"Error rate: {round(self.incorrect_comb / (total), 4)}")
        print(f"Average number of candidates: {round(self.can_comb_len / total, 2)}")

    def merge(self, fingerprinting_results, context_results, test_df):
        with Logger() as logger:
            if len(fingerprinting_results) != len(context_results):
                logger.error("Results are not the same length.")

            for index, row in test_df.iterrows():
                # Get candidates from fingerprinting and context matching
                ja_comb_candidates = fingerprinting_results[index][
                    "combined_candidates"
                ]
                ja_candidates = fingerprinting_results[index]["ja_candidates"]
                # Get candidates from context matching.
                context_candidates = context_results[index]

                logger.debug(
                    f"{row[APP_NAME]}:{context_candidates}:{ja_comb_candidates}"
                )

                final_candidates = ja_candidates.union(context_candidates)
                final_comb_candidates = ja_comb_candidates.union(context_candidates)

                self.can_len += len(final_candidates)
                if row[APP_NAME] in final_candidates:
                    self.correct += 1
                else:
                    self.incorrect += 1

                self.can_comb_len += len(final_comb_candidates)
                if row[APP_NAME] in final_comb_candidates:
                    self.correct_comb += 1
                else:
                    self.incorrect_comb += 1

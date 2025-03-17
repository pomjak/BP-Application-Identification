"""
File: config.py
Description: This file contains methods for identification of applications using ja/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 17/03/2025
"""

from constants import get_keys, APP_NAME
from database import Database
from logger import Logger


class FingerprintingMethod:
    def __init__(self, version):
        self.JA_key, self.JAS_key, self.SNI_key = get_keys(version)
        with Logger() as logger:
            logger.info(f"JA version set: {version}")
            logger.debug(f"JA key: {self.JA_key}")
            logger.debug(f"JAS key: {self.JAS_key}")
            logger.debug(f"SNI key: {self.SNI_key}")
        self.correct = 0
        self.incorrect = 0
        self.correct_combination = 0
        self.incorrect_combination = 0
        self.len_candidates = 0
        self.len_candidates_combination = 0

    def __get_statistics(self):
        return (
            self.correct,
            self.incorrect,
            self.correct + self.incorrect,
            self.len_candidates,
        )

    def __get_statistics_combination(self):
        return (
            self.correct_combination,
            self.incorrect_combination,
            self.correct_combination + self.incorrect_combination,
            self.len_candidates_combination,
        )

    def display_statistics(self):
        correct, incorrect, total, len_cand = self.__get_statistics()
        print("JA fingerprinting:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct / total}")
        print(f"Average number of candidates: {round(len_cand / total, 2)}")

        correct, incorrect, total, len_cand_comb = self.__get_statistics_combination()
        print("________________________________________________________")
        print("Combination of JA + JAS + SNI")
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct / total}")
        print(f"Average number of candidates: {round(len_cand_comb / total, 2)}")

    def _resolve_and_update(self, appname, candidates):
        self.len_candidates += len(candidates)
        if appname in candidates:
            self.correct += 1
        else:
            self.incorrect += 1

    def _resolve_and_update_combination(self, appname, candidates):
        self.len_candidates_combination += len(candidates)
        if appname in candidates:
            self.correct_combination += 1
        else:
            self.incorrect_combination += 1

    def identify(self, db: Database, context=False):
        # iterate over test dataset and check if app name is in set of candidates
        for index, row in db.test_df.iterrows():
            # extract JA hash and app name from one row of ds
            ja = row[self.JA_key]
            jas = row[self.JAS_key]
            sni = row[self.SNI_key]
            appname = row[APP_NAME]

            # get sets of candidates for each fingerprint
            ja_candidates = db.get_app(self.JA_key, ja)
            jas_candidates = db.get_app(self.JAS_key, jas)
            sni_candidates = db.get_app(self.SNI_key, sni)

            # filter out empty sets
            non_empty_sets = [
                candidates
                for candidates in [ja_candidates, jas_candidates, sni_candidates]
                if candidates
            ]

            # intersect all not-empty sets
            if non_empty_sets:
                candidates = set.intersection(*non_empty_sets)
            else:
                candidates = set()

            # check if candidates match real app name and update statistics accordingly
            self._resolve_and_update(appname, ja_candidates)
            self._resolve_and_update_combination(appname, candidates)
            db.fingerprinting_results[index] = {
                "ja_candidates": frozenset(ja_candidates),
                "combined_candidates": frozenset(candidates),
            }

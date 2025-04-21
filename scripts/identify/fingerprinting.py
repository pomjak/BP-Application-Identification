"""
File: fingerpinting.py
Description: This file contains methods for identification of applications using ja3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 21/04/2025
"""

from config import get_keys, APP_NAME
from .database import Database
from .logger import Logger

import numpy as np


class FingerprintingMethod:
    def __init__(self, version):
        self.version = version
        self.JA_key, self.JAS_key, self.SNI_key = get_keys(version)
        with Logger() as logger:
            logger.info(f"Selecting JA{version} version")
            logger.debug(f"JA key: {self.JA_key}")
            logger.debug(f"JAS key: {self.JAS_key}")
            logger.debug(f"SNI key: {self.SNI_key}")
        self.correct = 0
        self.incorrect = 0
        self.correct_combination = 0
        self.incorrect_combination = 0
        self.len_candidates = []
        self.len_candidates_combination = []

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
        print(f"JA{self.version} fingerprinting:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {round((correct / total), 4)}")
        avg_len = sum(len_cand) / len(len_cand)
        median_len = np.median(len_cand)
        modus_len = max(set(len_cand), key=len_cand.count)
        print(f"Average len of candidates: {round(avg_len, 4)}")
        print(f"Median len of candidates: {round(median_len, 4)}")
        print(f"Modus len of candidates: {round(modus_len, 4)}")
        print(f"Max len of candidates: {max(len_cand)}")
        print(f"Min len of candidates: {min(len_cand)}\n")

        correct, incorrect, total, len_cand_comb = self.__get_statistics_combination()
        print("________________________________________________________")
        print(f"Combination of JA{self.version} + JA{self.version}S + SNI")
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {round((correct / total), 4)}")

        avg_len = sum(len_cand_comb) / len(len_cand_comb)
        median_len = np.median(len_cand_comb)
        modus_len = max(set(len_cand_comb), key=len_cand_comb.count)
        print(f"Average len of candidates: {round(avg_len, 4)}")
        print(f"Median len of candidates: {round(median_len, 4)}")
        print(f"Modus len of candidates: {round(modus_len, 4)}")
        print(f"Max len of candidates: {max(len_cand_comb)}")
        print(f"Min len of candidates: {min(len_cand_comb)}\n")

    def _resolve_and_update(self, appname, candidates):
        self.len_candidates.append(len(candidates))
        if appname in candidates:
            self.correct += 1
        else:
            self.incorrect += 1

    def _resolve_and_update_combination(self, appname, candidates):
        self.len_candidates_combination.append(len(candidates))
        if appname in candidates:
            self.correct_combination += 1
        else:
            self.incorrect_combination += 1

    def get_ja_candidates(self, tls_entry, db: Database):
        # extract JA hash and app name from one row of ds
        ja = tls_entry[self.JA_key]

        # get sets of candidates for one fingerprint
        ja_candidates = db.get_app(self.JA_key, ja)
        return ja_candidates

    def get_ja_comb_candidates(self, tls_entry, db: Database, ja_candidates):
        # extract JA hash and app name from one row of ds
        jas = tls_entry[self.JAS_key]
        sni = tls_entry[self.SNI_key]

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

        return candidates

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info("Identifying using fingerprinting method...")
            # iterate over test dataset and check if app name is in set of candidates
            for index, row in db.test_df.iterrows():
                # get real app name
                appname = row[APP_NAME]

                # get sets of candidates for one fingerprint
                ja_candidates = self.get_ja_candidates(row, db)
                candidates = self.get_ja_comb_candidates(row, db, ja_candidates)

                # check if candidates match real app name and update statistics accordingly
                self._resolve_and_update(appname, ja_candidates)
                self._resolve_and_update_combination(appname, candidates)

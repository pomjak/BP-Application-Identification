"""
File: config.py
Description: This file contains methods for identification of applications using JA3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 19/02/2025
"""

import constants as col_names
from database import Database


class FingerprintingMethod:
    def __init__(self):
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
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct / total}")
        print(f"Average number of candidates: {round(len_cand / total, 2)}")

        correct, incorrect, total, len_cand_comb = self.__get_statistics_combination()
        print("________________________________________________________")
        print("combination of JA + JAS + SNI")
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

    def identify(self, db: Database):
        raise NotImplementedError("This method should be overridden by derived classes")


class JA3(FingerprintingMethod):
    def identify(self, db: Database):
        # iterate over test dataset and check if app name is in set of candidates
        for _, row in db.test_df.iterrows():
            # extract JA3 hash and app name from one row of ds
            ja3 = row[col_names.JA3]
            ja3s = row[col_names.JA3_S]
            sni = row[col_names.SNI]
            appname = row[col_names.APP_NAME]

            # get sets of candidates for each fingerprint
            ja3_candidates = db.get_app(col_names.JA3, ja3)
            ja3s_candidates = db.get_app(col_names.JA3_S, ja3s)
            sni_candidates = db.get_app(col_names.SNI, sni)

            # filter out empty sets
            non_empty_sets = [
                candidates
                for candidates in [ja3_candidates, ja3s_candidates, sni_candidates]
                if candidates
            ]

            # intersect all not-empty sets
            if non_empty_sets:
                candidates = set.intersection(*non_empty_sets)
            else:
                candidates = set()

            # check if candidates match real app name and update statistics accordingly
            self._resolve_and_update(appname, ja3_candidates)
            self._resolve_and_update_combination(appname, candidates)


class JA4(FingerprintingMethod):
    def identify(self, db: Database):
        for _, row in db.test_df.iterrows():
            ja4 = row[col_names.JA4]
            ja4s = row[col_names.JA4_S]
            sni = row[col_names.SNI]
            appname = row[col_names.APP_NAME]

            ja4_candidates = db.get_app(col_names.JA4, ja4)
            ja4s_candidates = db.get_app(col_names.JA4_S, ja4s)
            sni_candidates = db.get_app(col_names.SNI, sni)

            non_empty_sets = [
                candidates
                for candidates in [ja4_candidates, ja4s_candidates, sni_candidates]
                if candidates
            ]
            if non_empty_sets:
                candidates = set.intersection(*non_empty_sets)
            else:
                candidates = set()

            self._resolve_and_update(appname, ja4_candidates)

            self._resolve_and_update_combination(appname, candidates)

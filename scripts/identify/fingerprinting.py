"""
File: config.py
Description: This file contains methods for identification of applications using JA3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 18/11/2024
"""

import constants as col_names
from database import Database


class FingerprintingMethod:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.correct_full = 0
        self.incorrect_full = 0

    def statistics(self):
        return self.correct, self.incorrect, self.correct + self.incorrect

    def full_statistics(self):
        return self.correct_full, self.incorrect_full, self.correct_full + self.incorrect_full

    def display_statistics(self):
        correct, incorrect, total = self.statistics()
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct/total}")
        
        correct, incorrect, total = self.full_statistics()
        print("________________________________________________________")
        print("combination of JA + JAS + SNI")
        print("Real app name was found in set of candidates:")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {correct/total}")

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

            ja3_candidates = db.get_app(col_names.JA3, ja3)
            ja3s_candidates = db.get_app(col_names.JA3_S, ja3s)
            sni_candidates = db.get_app(col_names.SNI, sni)

            candidates = ja3_candidates.union(ja3s_candidates).union(sni_candidates)

            if appname in ja3_candidates:
                self.correct += 1
            else:
                self.incorrect += 1
            
            if appname in candidates:
                self.correct_full += 1
            else:
                self.incorrect_full += 1


class JA4(FingerprintingMethod):
    def identify(self, db: Database):
        for _, row in db.test_df.iterrows():
            ja4= row[col_names.JA4]
            ja4s = row[col_names.JA4_S]
            sni = row[col_names.SNI]
            appname = row[col_names.APP_NAME]
            
            ja4_candidates = db.get_app(col_names.JA4,ja4)
            ja4s_candidates = db.get_app(col_names.JA4_S,ja4s)
            sni_candidates = db.get_app(col_names.SNI,sni)
            
            candidates = ja4_candidates.union(ja4s_candidates).union(sni_candidates)
            
            if appname in ja4_candidates:
                self.correct += 1
            else:
                self.incorrect += 1
                
            if appname in candidates:
                self.correct_full += 1
            else:
                self.incorrect_full += 1

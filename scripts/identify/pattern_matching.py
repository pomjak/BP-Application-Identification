"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 01/03/2025
"""

from prefixspan import prefixspan
from database import Database
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# from spade import spade as sp

import constants as col_names


class PatternMatchingMethod:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def display_statistics(self):
        pass

    def identify(self, df):
        raise NotImplementedError("This method should be overridden by subclasses")


class Apriori(PatternMatchingMethod):

    def train(self, db: Database):
        pass

    def identify(self, db):
        pass


class PrefixSpan(PatternMatchingMethod):
    def __init__(self):
        pass

    def train(self, db):
        pass

    def identify(self, db):
        pass


class SPADE(PatternMatchingMethod):
    def __init__(self):
        pass

    def train(self, db):
        pass

    def identify(self, db):
        pass

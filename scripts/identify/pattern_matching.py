"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 14/01/2025
"""

from prefixspan import prefixspan
from mlxtend.frequent_patterns import apriori

# from spade import spade as sp


class PatternMatchingMethod:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def display_statistics(self):
        pass

    def identify(self, df):
        raise NotImplementedError("This method should be overridden by subclasses")


class Apriori(PatternMatchingMethod):
    def __init__(self):
        pass
    
    def train(self, df):
        pass
    
    def identify(self, df):
        pass


class PrefixSpan(PatternMatchingMethod):
    def __init__(self):
        pass

    def identify(self, data):
        pass


class SPADE(PatternMatchingMethod):
    def __init__(self):
        pass

    def identify(self, data):
        pass

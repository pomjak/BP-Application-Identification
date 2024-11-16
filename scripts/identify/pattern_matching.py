"""
File: pattern_matching.py
Description: This file contains algorithms for detecting frequent patterns.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 16/11/2024
"""

from prefixspan import prefixspan as ps
from mlxtend.frequent_patterns import apriori as ap

# from spade import spade as sp


class PatternMatchingMethod:
    def find_patterns(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")


class Apriori(PatternMatchingMethod):
    def find_patterns(self, data):
        raise NotImplementedError("Apriori algo called")


class PrefixSpan(PatternMatchingMethod):
    def find_patterns(self, data):
        raise NotImplementedError("PrefixSpan algo called")


class SPADE(PatternMatchingMethod):
    def find_patterns(self, data):
        raise NotImplementedError("SPADE algo called")

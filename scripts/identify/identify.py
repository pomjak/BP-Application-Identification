"""
File: identify.py
Description: Main file for identification of applications using JA3/4 fingerprinting and frequent pattern matching algorithms.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 16/11/2024
"""

from config import config
from fingerprinting import JA3, JA4
from pattern_matching import Apriori, SPADE, PrefixSpan


def main():
    dataset_path = config.ds
    fingerprint_method = config.ja
    pattern_algo = config.pattern_algo

    print(f"Dataset path: {dataset_path}")
    print(f"Fingerprinting method: {fingerprint_method}")
    print(f"Pattern matching algorithm: {pattern_algo}")


if __name__ == "__main__":
    main()

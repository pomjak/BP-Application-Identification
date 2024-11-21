"""
File: constants.py
Description: This file contains constants representing column names in the dataset.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 17/11/2024
Updated: 21/11/2024
"""

# modify the constants to match the dataset column names
JA3 = "JA3hash"
JA3_S = "JA3Shash"
JA4 = "JA4hash"
JA4_S = "JA4Shash"
APP_NAME = "AppName"
SNI = "SNI"
TYPE = "Version"


def get_keys(ja_version):
    # select correct col names based on version of JA
    return [JA4, JA4_S, SNI] if ja_version == 4 else [JA3, JA3_S, SNI]

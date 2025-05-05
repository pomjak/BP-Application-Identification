"""
File: config.py
Description: This file contains constants representing column names in the dataset, other constants and setting .
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 17/11/2024
Updated: 05/05/2025
"""

# Modify the constants to match the dataset column names
JA3 = "JA3hash"
JA3_S = "JA3Shash"
JA4 = "JA4hash"
JA4_S = "JA4Shash"
APP_NAME = "AppName"
SNI = "SNI"
TYPE = "Type"
FILE = "Filename"
ORG = "OrgName"

TLS_VERSION = "TLSVersion"
CIPHER_SUITE = "ClientCipherSuite"
CLIENT_EXT = "ClientExtensions"
CLIENT_SUPPORTED_GROUPS = "ClientSupportedGroups"
CLIENT_SUPPORTED_VERSIONS = "ClientSupportedVersions"
EC_FMT = "EC_fmt"
ALPN = "ALPN"
SIGNATURE_ALGORITHMS = "SignatureAlgorithms"
SERVER_CIPHER_SUITE = "ServerCipherSuite"
SERVER_EXTENSIONS = "ServerExtensions"
SERVER_SUPPORTED_VERSIONS = "ServerSupportedVersions"


# Select which columns  to keep in db for fingerprinting methods and identification using context
columns_to_keep_in_db = [
    APP_NAME,  # need to be kept, excluded from identification
    FILE,  # need to be kept, excluded from identification
    JA3,  # kept for fingerprinting, in each run is selected only one version (JA3/JA4)
    JA3_S,
    JA4,
    JA4_S,
    SNI,  # needed for combination
    #! FROM HERE INSERT EVERY ATTRIBUTE USED LATER FOR CONTEXT IDENTIFICATION
    ORG,
    # TLS_VERSION,
    # CIPHER_SUITE,
    # CLIENT_EXT,
    # CLIENT_SUPPORTED_GROUPS,
    # CLIENT_SUPPORTED_VERSIONS,
    # EC_FMT,
    # ALPN,
    # SIGNATURE_ALGORITHMS,
    # SERVER_CIPHER_SUITE,
    # SERVER_EXTENSIONS,
    # SERVER_SUPPORTED_VERSIONS,
]


#! INSERT HERE ATTRIBUTES FOR CONTEXT IDENTIFICATION
columns_to_keep_for_context = [
    # JA3,
    # JA3_S,
    JA4,
    JA4_S,
    # SNI,
    # ORG,
    # TLS_VERSION,
    # CIPHER_SUITE,
    # CLIENT_EXT,
    # CLIENT_SUPPORTED_GROUPS,
    # CLIENT_SUPPORTED_VERSIONS,
    # EC_FMT,
    # ALPN,
    # SIGNATURE_ALGORITHMS,
    # SERVER_CIPHER_SUITE,
    # SERVER_EXTENSIONS,
    # SERVER_SUPPORTED_VERSIONS,
]

#! EDIT LENGTH AND COUNT OF PATTERNS STORED PER ONE APP
PATTERN_FILTERS = [
    {"operator": "==", "length": 2, "head": 10},
    # {"operator": "==", "length": 3, "head": 10},
    # {"operator": "==", "length": 4, "head": 10},
]


# DEBUG LOG LEVEL
DEBUG_ENABLED = True


def get_keys(ja_version):
    # select correct col names based on version of JA
    return [JA4, JA4_S, SNI] if ja_version == 4 else [JA3, JA3_S, SNI]

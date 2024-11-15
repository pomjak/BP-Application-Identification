"""
File: config.py
Description: This file contains methods for identification of applications using JA3/4 fingerprinting.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
"""
class FingerprintingMethod:
    def fingerprint(self, data):
        raise NotImplementedError("This method should be overridden by derived classes")

class JA3(FingerprintingMethod):
    def fingerprint(self, data):
        raise NotImplementedError("JA3 fingerprint method called")

class JA4(FingerprintingMethod):
    def fingerprint(self, data):
        raise NotImplementedError("JA4 fingerprint method called")

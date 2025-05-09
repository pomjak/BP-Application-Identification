"""
File: ja_context.py
Description: This file contains the JA_Context class that implements the JA3/4 fingerprinting method with context-aware identification.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 09/05/2025
"""

import config as CONFIG
from .database import Database
from .fingerprinting import FingerprintingMethod
from .pattern_matching import Apriori
from .logger import Logger

import numpy as np
import pandas as pd


class JA_Context:
    def __init__(
        self,
        fingerprinting: FingerprintingMethod,
        context: Apriori,
        sliding_window_size,
    ):
        self.context = context
        self.fingerprinting = fingerprinting
        self.sliding_window_size = sliding_window_size

    def shuffle_df(self, df):
        grouped_by_file = df.groupby(CONFIG.FILE)
        grouped_by_app = df.groupby(CONFIG.APP_NAME)

        with Logger() as logger:
            logger.info(
                f"Average length of each group: {grouped_by_file.size().mean()}"
            )
            self.context.group_size_mean = grouped_by_file.size().mean()

        lookup = {
            app: list(group[CONFIG.FILE].unique()) for app, group in grouped_by_app
        }

        # Distribute files while ensuring no adjacent app duplicates.
        shuffled_filenames = []
        while any(lookup.values()):  # While there are still files left
            for app in lookup:  # Iterate through apps in a fixed order
                if lookup[app]:  # If there are files left for this app
                    shuffled_filenames.append(lookup[app].pop(0))  # Take one file

        # Reconstruct DataFrame in the new order
        return pd.concat(
            [grouped_by_file.get_group(fname) for fname in shuffled_filenames]
        ).reset_index(drop=True)

    def _log_apps_in_window(self, window):
        with Logger() as logger:
            files_in_window = window.groupby(CONFIG.FILE)
            apps_in_window = []
            for _, file in files_in_window:
                apps_in_window.append(file[CONFIG.APP_NAME].iloc[0])
            logger.debug(f"Apps in window: {apps_in_window}")

    def identify(self, db: Database):
        with Logger() as logger:
            self._log_identification_start()

            test_df = self._prepare_test_data(db)
            num_test_launches = len(test_df)
            self.context.number_of_tls = num_test_launches

            logger.info(
                f"Sliding window size: {self.sliding_window_size}, "
                f"number of test launches: {num_test_launches}"
            )
            self.context.sliding_window_size = self.sliding_window_size
            for i in range(num_test_launches):
                self._process_window(i, test_df, db)

    def _log_identification_start(self):
        with Logger() as logger:
            logger.info(
                "Identifying applications using JA3/4 fingerprints and context..."
            )

    def _prepare_test_data(self, db: Database):
        test_df = db.get_test_df()
        return self.shuffle_df(test_df)

    def _slide_window(self, index: int, test_df):
        window_size = self.sliding_window_size
        half_window = window_size // 2
        num_test_launches = len(test_df)
        # Ensure the window is centered around the current index.
        window_start = np.clip(index - half_window, 0, num_test_launches - window_size)
        row_index_within_window = index - window_start
        # Ensure the row index is within the bounds of the window.
        window = test_df.iloc[window_start : window_start + window_size]
        row = window.iloc[row_index_within_window]

        with Logger() as logger:
            logger.debug(
                f"Window position: start={window_start}, end={window_start + window_size - 1}"
            )
            logger.debug(f"Row index within window: {row_index_within_window}")
            logger.debug(f"Real app: {row[CONFIG.APP_NAME]}")

        return window, row

    def _process_window(self, index: int, test_df, db: Database):
        window, row = self._slide_window(index, test_df)
        real_app = row[CONFIG.APP_NAME]

        self._log_apps_in_window(window)

        ja_candidates = self._get_ja_candidates(row, db)
        ja_comb_candidates = self._get_ja_comb_candidates(row, db, ja_candidates)

        self._evaluate_context_and_update_stats(
            db, window, real_app, ja_candidates, ja_comb_candidates
        )

    def _get_ja_candidates(self, row, db):
        with Logger() as logger:
            candidates = self.fingerprinting.get_ja_candidates(row, db)
            if not candidates:
                logger.warn("Empty JA candidates")
                self.context.empty_ja += 1
            else:
                logger.debug(f"JA: {candidates}")
            return candidates

    def _get_ja_comb_candidates(self, row, db, ja_candidates):
        # Wrapper for the fingerprinting get_ja_comb_candidates method
        with Logger() as logger:
            candidates = self.fingerprinting.get_ja_comb_candidates(
                row, db, ja_candidates
            )
            if not candidates:
                logger.warn("Empty JA_comb candidates")
                self.context.empty_ja_comb += 1
            else:
                logger.debug(f"JA COMB: {candidates}")
            return candidates

    def _evaluate_context_and_update_stats(
        self, db, window, real_app, ja_candidates, ja_comb_candidates
    ):
        with Logger() as logger:
            db_for_ja = self._filter_frequent_patterns(db, ja_candidates)
            db_for_ja_comb = self._filter_frequent_patterns(db, ja_comb_candidates)

            ja_context = self._find_context_candidates(
                db_for_ja, db, window, is_comb=False
            )
            logger.debug(f"CONTEXT (JA) : {[app for (app, score) in ja_context]}")
            self.context._update_statistics(real_app, ja_context, is_comb=False)

            ja_comb_context = self._find_context_candidates(
                db_for_ja_comb, db, window, is_comb=True
            )
            logger.debug(
                f"CONTEXT (JA COMB): {[app for (app, score) in ja_comb_context]}\n"
            )
            self.context._update_statistics(real_app, ja_comb_context, is_comb=True)

    def _filter_frequent_patterns(self, db, candidates):
        if not candidates:
            return {}
        # Filter candidates to only include those present in the database
        return {
            key: db.frequent_patterns[key]
            for key in candidates
            if key in db.frequent_patterns
        }

    def _find_context_candidates(self, db_subset, db, window, is_comb):
        """finds candidates using context

        Args:
            db_subset (dict): Subset of the database to search for candidates.
            db (dict): Database containing the frequent patterns.
            window (df): Sliding window of data to analyze.
            is_comb (bool): Flag indicating if the context is for a combination of fingerprints.

        Returns:
            list: of top N candidates found using patterns and shortened database.
        """
        with Logger() as logger:
            if not db_subset:
                self._log_empty_subset(logger, is_comb)
                return self.context.find_similarity(db.frequent_patterns, window)

            candidates = self.context.find_similarity(db_subset, window)

            if not candidates:
                logger.info("No candidates found. Falling back to pure context.")
                return self._use_pure_context(
                    db.frequent_patterns, db_subset, window, is_comb
                )

            return candidates

    def _log_empty_subset(self, logger, is_comb):
        context_label = "[comb]" if is_comb else ""
        logger.info(
            f"Subset of DB is empty, using whole database for context. {context_label}"
        )

        if is_comb:
            self.context.context_using_whole_db_comb += 1
        else:
            self.context.context_using_whole_db += 1

    def _use_pure_context(self, patterns, db_subset, window, is_comb):
        with Logger() as logger:
            context_type = "[comb]" if is_comb else ""
            logger.info(
                f"Failed to find similarity with subset db. Falling back to pure context using complement of db. {context_type}"
            )

            self._increment_pure_context_counter(is_comb)

            logger.debug(f"Database subset keys: {list(db_subset.keys())}")

            db_complement = self._get_db_complement(patterns, db_subset)
            logger.debug(f"Database complement keys: {list(db_complement.keys())}")

            candidates = self.context.find_similarity(db_complement, window)

            if not candidates:
                logger.info("No candidates found in complement. Using full patterns.")
                candidates = self.context.find_similarity(patterns, window)

            return candidates

    def _increment_pure_context_counter(self, is_comb):
        if is_comb:
            self.context.pure_context_comb += 1
        else:
            self.context.pure_context += 1

    def _get_db_complement(self, patterns, db_subset):
        return {key: value for key, value in patterns.items() if key not in db_subset}

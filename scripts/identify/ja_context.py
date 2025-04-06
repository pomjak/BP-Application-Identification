from database import Database
import constants as Constants
from pattern_matching import PatternMatchingMethod
import numpy as np
import pandas as pd
from logger import Logger


class JA_Context(PatternMatchingMethod):
    def __init__(
        self,
        fingerprinting,
        context,
        sliding_window_size,
        max_candidates_size,
    ):
        super().__init__(
            0.15,
            context.ja_version,
            max_candidates_size,
        )

        self.context = context
        self.fingerprinting = fingerprinting
        self.sliding_window_size = sliding_window_size

    def shuffle_df(self, df):
        grouped_by_file = df.groupby(Constants.FILE)
        grouped_by_app = df.groupby(Constants.APP_NAME)

        with Logger() as logger:
            logger.info(
                f"Average length of each group: {grouped_by_file.size().mean()}"
            )

            logger.info(f"Modus of group sizes: {grouped_by_file.size().mode()[0]}")

        lookup = {
            app: list(group[Constants.FILE].unique()) for app, group in grouped_by_app
        }

        # Distribute files while ensuring no adjacent app duplicates
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
            files_in_window = window.groupby(Constants.FILE)
            apps_in_window = []
            for _, file in files_in_window:
                apps_in_window.append(file[Constants.APP_NAME].iloc[0])
            logger.debug(f"Apps in window: {apps_in_window}")

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info(
                "Identifying applications using JA3/4 fingerprints and context..."
            )

            test_df = db.get_test_df()
            test_df = self.shuffle_df(test_df)

            window_size = self.sliding_window_size
            num_test_launches = len(test_df)
            self.number_of_tls = num_test_launches

            logger.info(
                f"Sliding window size: {window_size}, number of test launches: {num_test_launches}"
            )

            half_window = window_size // 2

            for i in range(num_test_launches):
                window_start = np.clip(
                    i - half_window, 0, num_test_launches - window_size
                )
                row_index_within_window = i - window_start

                window = test_df.iloc[window_start : window_start + window_size]
                row = window.iloc[row_index_within_window]
                real_app = row[Constants.APP_NAME]
                logger.debug(
                    f"Window position: start={window_start}, end={window_start + window_size - 1}"
                )
                logger.debug(f"Row index within window: {row_index_within_window}")
                logger.debug(f"Real app: {real_app}")

                self._log_apps_in_window(window)

                ja_candidates = self.fingerprinting.get_ja_candidates(row, db)

                if not ja_candidates:
                    logger.warn("Empty JA candidates")
                    self.empty_ja += 1
                else:
                    logger.debug(f"JA: {ja_candidates}")

                ja_comb_candidates = self.fingerprinting.get_ja_comb_candidates(
                    row, db, ja_candidates
                )
                if not ja_comb_candidates:
                    logger.warn("Empty JA_comb candidates")
                    self.empty_ja_comb += 1
                else:
                    logger.debug(f"JA COMB: {ja_comb_candidates}")

                db_for_ja = self._filter_frequent_patterns(db, ja_candidates)
                db_for_ja_comb = self._filter_frequent_patterns(db, ja_comb_candidates)

                ja_context_candidates = self._find_context_candidates(
                    db_for_ja, db, window, is_comb=False
                )
                logger.debug(
                    f"CONTEXT (JA) : {[app for (app, score) in ja_context_candidates]}"
                )

                ja_comb_context_candidates = self._find_context_candidates(
                    db_for_ja_comb, db, window, is_comb=True
                )
                logger.debug(
                    f"CONTEXT (JA COMB): {[app for (app, score) in ja_comb_context_candidates]}\n"
                )

                self._update_statistics(real_app, ja_context_candidates, is_comb=False)
                self._update_statistics(
                    real_app, ja_comb_context_candidates, is_comb=True
                )

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
        with Logger() as logger:
            if not db_subset:
                logger.info(
                    f"Subset of db is empty {db_subset}, using whole database for context.{'[comb]' if is_comb else ''}"
                )
                if is_comb:
                    self.context_using_whole_db_comb += 1
                else:
                    self.context_using_whole_db += 1
                return self.context.find_similarity(db.frequent_patterns, window)

            candidates = self.context.find_similarity(db_subset, window)

            if not candidates:
                logger.info("Candidates is empty. Falling back to pure context.")
                # Fallback to pure context if no candidates are found
                return self._use_pure_context(
                    db.frequent_patterns, db_subset, window, is_comb
                )
            return candidates

    def _use_pure_context(self, patterns, db_subset, window, is_comb):
        with Logger() as logger:
            logger.info(
                f"Failed finding similarity with subset db. Using pure context with complement of db.{'[comb]' if is_comb else ''}"
            )
            logger.debug(f"Database subset: {db_subset.keys()}")
            db_complement = {
                key: value for key, value in patterns.items() if key not in db_subset
            }
            logger.debug(f"Database complement: {db_complement.keys()}")
            candidates = self.context.find_similarity(db_complement, window)
            if is_comb:
                self.pure_context_comb += 1
            else:
                self.pure_context += 1
            if not candidates:
                candidates = self.context.find_similarity(patterns, window)
        return candidates

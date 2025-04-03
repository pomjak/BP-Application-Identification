from database import Database
import constants as Constants
from pattern_matching import PatternMatchingMethod
import numpy as np
import pandas as pd
from logger import Logger


class JA_Context(PatternMatchingMethod):
    def __init__(self, fingerprinting, context, sliding_window_size):
        super().__init__(0.15, context.ja_version)

        self.context = context
        self.fingerprinting = fingerprinting
        self.sliding_window_size = sliding_window_size

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info(
                "Identifying applications using JA3/4 fingerprints and context..."
            )

            test_df = db.get_test_df()
            grouped = test_df.groupby(Constants.FILE)
            shuffled_filenames = np.random.permutation(test_df[Constants.FILE].unique())

            logger.info(f"Average length of each group: {grouped.size().mean()}")

            shuffled_test_df = pd.concat(
                [grouped.get_group(fname) for fname in shuffled_filenames]
            ).reset_index(drop=True)

            window_size = self.sliding_window_size
            num_test_launches = len(shuffled_test_df)
            self.number_of_tls = num_test_launches

            logger.info(
                f"Sliding window size: {window_size}, number of test launches: {num_test_launches}"
            )

            half_window = window_size // 2

            for i in range(num_test_launches):
                if i < half_window:
                    # Fix window at the start, slide row in center of window
                    window_start = 0
                    row_index = i
                elif i >= num_test_launches - half_window:
                    # Fix window at the end, slide row till the end
                    window_start = num_test_launches - window_size
                    row_index = window_size - (num_test_launches - i)
                else:
                    # Normal sliding window behavior
                    window_start = i - half_window
                    row_index = half_window

                window = shuffled_test_df.iloc[
                    window_start : window_start + window_size
                ]
                row = window.iloc[row_index]
                real_app = row[Constants.APP_NAME]

                ja_candidates = self.fingerprinting.get_ja_candidates(row, db)
                self.empty_ja += 1 if ja_candidates is None else 0

                ja_comb_candidates = self.fingerprinting.get_ja_comb_candidates(
                    row, db, ja_candidates
                )
                self.empty_ja_comb += 1 if ja_comb_candidates is None else 0

                db_for_ja = self._filter_frequent_patterns(db, ja_candidates)
                db_for_ja_comb = self._filter_frequent_patterns(db, ja_comb_candidates)

                ja_context_candidates = self._find_context_candidates(
                    db_for_ja, db, window, is_comb=False
                )
                ja_comb_context_candidates = self._find_context_candidates(
                    db_for_ja_comb, db, window, is_comb=True
                )

                self._update_statistics(real_app, ja_context_candidates, is_comb=False)
                self._update_statistics(
                    real_app, ja_comb_context_candidates, is_comb=True
                )

    def _filter_frequent_patterns(self, db, candidates):
        return {
            key: db.frequent_patterns[key]
            for key in candidates
            if key in db.frequent_patterns
        }

    def _find_context_candidates(self, db_subset, db, window, is_comb):
        candidates = self.context.find_similarity(db_subset, window)
        if not candidates:
            candidates = self._use_pure_context(
                db.frequent_patterns, db_subset, window, is_comb
            )
        return candidates

    def _use_pure_context(self, patterns, db_subset, window, is_comb):
        db_complement = {
            key: value for key, value in patterns.items() if key not in db_subset
        }
        candidates = self.context.find_similarity(db_complement, window)
        if is_comb:
            self.pure_context_comb += 1
        else:
            self.pure_context += 1
        return candidates

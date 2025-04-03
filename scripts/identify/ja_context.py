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

        # Step 1: Create a lookup dictionary with app names and their files
        lookup = {
            app: list(group[Constants.FILE].unique()) for app, group in grouped_by_app
        }

        # Step 2: Sort apps for deterministic order

        # Step 3: Distribute files while ensuring no adjacent app duplicates
        shuffled_filenames = []
        while any(lookup.values()):  # While there are still files left
            for app in lookup:  # Iterate through apps in a fixed order
                if lookup[app]:  # If there are files left for this app
                    shuffled_filenames.append(lookup[app].pop(0))  # Take one file

        # Step 4: Reconstruct DataFrame in the new order
        return pd.concat(
            [grouped_by_file.get_group(fname) for fname in shuffled_filenames]
        ).reset_index(drop=True)

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
                row_index = i - window_start

                window = test_df.iloc[window_start : window_start + window_size]
                row = window.iloc[row_index]
                real_app = row[Constants.APP_NAME]

                window = test_df.iloc[window_start : window_start + window_size]
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

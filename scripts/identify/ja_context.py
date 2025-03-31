from database import Database
import constants as Constants
from pattern_matching import PatternMatchingMethod
import numpy as np
import pandas as pd
from logger import Logger


class JA_Context(PatternMatchingMethod):
    def __init__(self, fingerprinting, context, sliding_window_size):
        super().__init__(0.0)

        self.fingerprinting = fingerprinting
        self.context = context
        self.sliding_window_size = sliding_window_size

    def identify(self, db: Database):
        with Logger() as logger:
            logger.info(
                "Identifying applications using JA3/4 fingerprints and context..."
            )
            test_df = db.get_test_df()
            # Group rows by filename
            grouped = test_df.groupby(Constants.FILE)
            # Get unique filenames and shuffle them
            shuffled_filenames = np.random.permutation(test_df[Constants.FILE].unique())

            # print average length of each group
            avg_lengths = grouped.size().mean()
            logger.info(f"Average length of each group: {avg_lengths}")
            # Rebuild dataframe while keeping launches within a filename in order
            shuffled_test_df = pd.concat(
                [grouped.get_group(fname) for fname in shuffled_filenames]
            )
            # Reset index to ensure proper iteration
            shuffled_test_df = shuffled_test_df.reset_index(drop=True)
            # sliding window over testing dataset
            window_size = self.sliding_window_size
            logger.info(
                f"Sliding window size: {window_size}, number of test launches: {len(shuffled_test_df)}"
            )
            self.number_of_tls = len(test_df) - window_size + 1
            for i in range(self.number_of_tls):
                window = shuffled_test_df.iloc[i : i + window_size]
                row = window.iloc[window_size // 2]

                real_app = row[Constants.APP_NAME]
                apps_in_window = window[Constants.APP_NAME].unique()
                print(apps_in_window, real_app)
                ja_candidates = self.fingerprinting.get_ja_candidates(row, db)
                ja_combination_candidates = self.fingerprinting.get_ja_comb_candidates(
                    row, db, ja_candidates
                )

                db_for_ja = {
                    key: db.frequent_patterns[key]
                    for key in ja_candidates
                    if key in db.frequent_patterns
                }
                db_for_ja_comb = {
                    key: db.frequent_patterns[key]
                    for key in ja_combination_candidates
                    if key in db.frequent_patterns
                }

                ja_context_candidates = self.context.find_similarity(db_for_ja, window)

                ja_comb_context_candidates = self.context.find_similarity(
                    db_for_ja_comb, window
                )

                self._update_statistics(real_app, ja_context_candidates, is_comb=False)
                self._update_statistics(
                    real_app, ja_comb_context_candidates, is_comb=True
                )

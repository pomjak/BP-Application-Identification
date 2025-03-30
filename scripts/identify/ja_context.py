from database import Database
import constants as Constants
from pattern_matching import PatternMatchingMethod
import numpy as np
import pandas as pd


class JA_Context(PatternMatchingMethod):
    def __init__(self, fingerprinting, context):
        super().__init__()

        self.fingerprinting = fingerprinting
        self.context = context

    def identify(self, db: Database):
        test_df = db.get_test_df()
        # Group rows by filename
        grouped = test_df.groupby(Constants.FILE)
        # Get unique filenames and shuffle them
        shuffled_filenames = np.random.permutation(test_df[Constants.FILE].unique())

        # Rebuild dataframe while keeping launches within a filename in order
        shuffled_test_df = pd.concat(
            [grouped.get_group(fname) for fname in shuffled_filenames]
        )

        # Reset index to ensure proper iteration
        shuffled_test_df = shuffled_test_df.reset_index(drop=True)
        # sliding window over testing dataset
        window_size = 15
        for i in range(len(test_df) - window_size + 1):
            window = test_df.iloc[i : i + window_size]
            row = window.iloc[0]
            real_app = row[Constants.APP_NAME]

            # print(
            #     row[Constants.FILE],
            #     row.name,
            # )
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

            self._update_statistics(real_app, ja_comb_context_candidates)
        self.uniq_count = self._get_number_of_unique_patterns_sets(db.frequent_patterns)
        # Count how often each set is used and its corresponding guess position.
        self._count_usage(db)

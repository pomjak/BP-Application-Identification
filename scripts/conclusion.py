import pandas as pd

# Load and label data
filters_df = pd.read_csv("outs/ex5-mobile-filters.csv", sep=";")
nofilters_df = pd.read_csv("outs/ex5-mobile-no-filters.csv", sep=";")
sup025_df = pd.read_csv("outs/ex5-mobile-02.csv", sep=";")
# print last 2 avg_len_of_candidates
print(
    filters_df[
        filters_df[
            "avg_len_of_candidates",
            "median_len_of_candidates",
            "modus_len_of_candidates",
            "max_len_of_candidates",
            "min_len_of_candidates",
        ]
    ].tail(2)
)
# Add setting labels
filters_df["setting"] = "sup_01_filters"
nofilters_df["setting"] = "sup_01"
sup025_df["setting"] = "sup_2"

# Combine all data
df = pd.concat([filters_df, nofilters_df, sup025_df], ignore_index=True)

# Filter to window size 3 only
df = df[df["sliding_window_size"] == 3]
df = df[df["candidate_size"].isin([1, 3, 5, 7, 9])]
# Group by setting
for setting, group in df.groupby("setting"):
    print(f"\n%% LaTeX Table for Setting: {setting}")

    # Build structured DataFrame
    pivot_df = group.pivot_table(
        index="candidate_size",
        columns="is_comb",
        values=["accuracy_overall", "time", "avg_len_of_candidates"],
        aggfunc="max",
    )

    # Rename columns for clarity
    pivot_df.columns = [
        f"{metric}_{'combined' if is_comb else 'ja4'}"
        for metric, is_comb in pivot_df.columns
    ]
    pivot_df = pivot_df.reset_index()
    # Reorder columns (optional)
    columns_order = [
        "candidate_size",
        "accuracy_overall_ja4",
        "time_ja4",
        "avg_len_of_candidates_ja4",
        "accuracy_overall_combined",
        "time_combined",
        "avg_len_of_candidates_combined",
    ]
    pivot_df = pivot_df[columns_order]
    # Přejmenování sloupců do češtiny
    pivot_df = pivot_df.rename(
        columns={
            "candidate_size": "Počet kandidátů",
            "accuracy_overall_ja4": "Přesnost",
            "time_ja4": "Čas [s]",
            "avg_len_of_candidates_ja4": "Délka",
            "accuracy_overall_combined": "Přesnost",
            "time_combined": "Čas [s]",
            "avg_len_of_candidates_combined": "Délka",
        }
    )

    latex_header = r"""
    \begin{table}[H]
    \centering
    \caption{Výsledky pro \texttt{mobile.csv} se šířkou okna 3}
    \label{tab:iscx}
    \begin{tabular}{rccc|ccc}
    \toprule
    \multicolumn{7}{c}{\texttt{mobile.csv}}  \\
    \midrule
    \multirow{2}{*}{Počet kandidátů} & \multicolumn{3}{c}{JA4} & \multicolumn{3}{c}{JA4+JA4S+SNI}\\
    & Přesnost & Čas [s] & Prům. délka & Přesnost & Čas [s] & Prům. délka \\
    """

    # Vygeneruj tělo tabulky bez záhlaví
    latex_body = pivot_df.to_latex(
        index=False, header=False, float_format="%.3f", column_format="rccc|ccc"
    )

    latex_footer = r"""
    \end{table}
    """

    # Spojení hlavičky, těla a patičky
    full_latex_table = latex_header + latex_body + latex_footer

    # Výpis do konzole nebo zápis do souboru
    print(full_latex_table)

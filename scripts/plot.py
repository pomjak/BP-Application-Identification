import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_iscx = pd.read_csv("outs/ex3-iscx.csv", sep=";")
df_mobile = pd.read_csv("outs/ex3-mobile.csv", sep=";")

# Datasets to process
datasets = {"ISCX": df_iscx, "Mobile": df_mobile}
# sns.set(style="whitegrid")
# plt.rcParams.update(
#     {
#         "font.size": 18,
#         "axes.titlesize": 20,
#         "axes.labelsize": 18,
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "legend.fontsize": 14,
#         "legend.title_fontsize": 16,
#     }
# )
# Loop through both datasets
for name, df in datasets.items():
    for is_comb_flag, label in [(True, "JA4+JA4S+SNI"), (False, "JA4")]:
        df_sub = df[df["is_comb"] == is_comb_flag]

        # Pivot for heatmap
        pivot = df_sub.pivot_table(
            index="pattern_filters", columns="min_support", values="accuracy_overall"
        )

        # Sort columns descending (optional)
        pivot = pivot[sorted(pivot.columns, reverse=True)]

        # Plot heatmap
        plt.figure(figsize=(12, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Přesnost"},
        )
        plt.title(f"{name}: Heatmapa přesnosti ({label})")
        plt.xlabel("Minimální podpora")
        plt.ylabel("Filtry vzorů")
        plt.tight_layout()
        # Save to file
        fname = f"../dias/ex3-{name.lower()}-heatmap-{label.replace(' ', '_')}.pdf"
        # plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.show()

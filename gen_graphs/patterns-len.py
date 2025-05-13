import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load and clean data
file_list = glob.glob("outs/patterns/filtered-mobile/*.csv")
df_list = [
    pd.read_csv(f, sep=";").assign(
        source_file=f.split("/")[-1]
        .replace(".csv", "")
        .replace("outs/patterns/", "")
        .replace("_patterns", "")
    )
    for f in file_list
]
all_df = pd.concat(df_list, ignore_index=True)
# Ensure 'pattern' column is clean and count elements in each pattern
all_df["pattern_length"] = all_df["itemsets"].apply(lambda x: len(str(x).split(",")))

support_thresholds = [0.01]

for threshold in support_thresholds:
    filtered_df = all_df[all_df["support"] >= threshold]

    # Count pattern lengths per app
    length_counts = (
        filtered_df.groupby(["source_file", "pattern_length"])
        .size()
        .unstack(fill_value=0)
    )

    # Sort apps by total patterns for consistent x-axis
    length_counts = length_counts.loc[
        length_counts.sum(axis=1).sort_values(ascending=False).index
    ]

    # Plot
    ax = length_counts.plot(kind="bar", stacked=True, figsize=(6, 6), colormap="Dark2")

    # Formatting
    plt.title(f"Rozložení délek vzorů na aplikaci (podpora ≥ {threshold})")
    plt.xlabel("Aplikace")
    plt.ylabel("Počet vzorů")
    if len(length_counts) > 20:
        tick_positions = range(0, len(length_counts.index), 4)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(length_counts.index[tick_positions], rotation=45, ha="right")
    else:
        ax.set_xticks(range(len(length_counts.index)))
        ax.set_xticklabels(length_counts.index, rotation=45, ha="right")

    plt.legend(title="Délka vzoru", loc="best")
    plt.tight_layout()
    # plt.savefig(
    #     f"../dias/pattern_lengths_{threshold}_{ds}.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.show()

    mean_support = (
        filtered_df.groupby("source_file")["support"]
        .mean()
        .reindex(length_counts.index)
    )

fig, ax1 = plt.subplots(figsize=(6, 6))

# Bar plot on primary y-axis
length_counts.plot(kind="bar", stacked=True, ax=ax1, colormap="Dark2")
ax1.set_xlabel("Aplikace")
ax1.set_ylabel("Počet vzorů", color="black")
ax1.tick_params(axis="y", labelcolor="black")

# X-axis tick formatting
if len(length_counts) > 20:
    tick_positions = range(0, len(length_counts.index), 4)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(length_counts.index[tick_positions], rotation=45, ha="right")
else:
    ax1.set_xticks(range(len(length_counts.index)))
    ax1.set_xticklabels(length_counts.index, rotation=45, ha="right")

# Line plot on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    range(len(mean_support)),
    mean_support.values,
    color="black",
    marker="o",
    linestyle="-",
    label="Průměrná podpora",
)
ax2.set_ylabel("Průměrná podpora", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# Titles and legends
plt.title(f"Rozložení délek vzorů a průměrná podpora (support ≥ {threshold})")
# Merge legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine and place in upper right of plot
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", title="Legenda")

# Merge legends from both axes

plt.tight_layout()
plt.savefig(
    f"../dias/pattern_lengths_filtered_{threshold}_mobile.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

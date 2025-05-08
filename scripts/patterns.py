import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load and clean data
for ds in ["mobile", "iscx"]:
    file_list = glob.glob(f"outs/patterns/{ds}/*.csv")
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

    # Support thresholds to analyze
    support_thresholds = [0.01, 0.05, 0.2, 0.25]

    # Generate one plot per threshold
    for threshold in support_thresholds:
        filtered_df = all_df[all_df["support"] >= threshold]
        patterns_per_app = (
            filtered_df["source_file"].value_counts().sort_values(ascending=False)
        )
        mean_support = (
            filtered_df.groupby("source_file")["support"]
            .mean()
            .reindex(patterns_per_app.index)
        )

        fig, ax1 = plt.subplots(figsize=(6, 6))

        # Bar plot for number of patterns
        bars = ax1.bar(
            patterns_per_app.index,
            patterns_per_app.values,
            color="blue",
            label="Počet vzorů",
        )
        ax1.set_ylabel("Počet vzorů")
        ax1.set_xlabel("Zkratka aplikace")
        ax1.tick_params(axis="y")

        # Line plot for mean support on secondary y-axis
        ax2 = ax1.twinx()
        line = ax2.plot(
            patterns_per_app.index,
            mean_support.values,
            color="red",
            marker="o",
            label="Průměrná podpora",
        )
        ax2.set_ylabel("Průměrná podpora")
        ax2.tick_params(axis="y")

        # Clean x-axis labels
        if len(patterns_per_app.index) > 20:
            tick_positions = range(0, len(patterns_per_app.index), 4)
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(
                patterns_per_app.index[tick_positions], rotation=45, ha="right"
            )
        else:
            ax1.set_xticks(range(len(patterns_per_app.index)))
            ax1.set_xticklabels(patterns_per_app.index, rotation=45, ha="right")

        # Titles and legends
        plt.title(f"Podpora ≥ {threshold}: Počet vzorů a průměrná podpora na aplikaci")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if threshold == 0.01 or threshold == 0.05:
            ax1.legend(
                lines + lines2,
                labels + labels2,
                loc="upper center",
            )
        else:
            ax1.legend(
                lines + lines2,
                labels + labels2,
                loc="upper left",
            )

        plt.tight_layout()
        plt.savefig(
            f"../dias/patterns_support_{threshold}_{ds}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )

        plt.close(fig)

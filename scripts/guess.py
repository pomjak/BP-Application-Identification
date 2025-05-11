import pandas as pd
import matplotlib.pyplot as plt


def plot_guess_accuracy(df):
    guess_cols = [f"guess_perc_{i}" for i in range(1, 10)]
    x = list(range(1, 10))  # guess positions

    # Plot setup
    plt.figure(figsize=(6, 6))

    # Loop over both values of is_comb
    for is_comb_value in [False, True]:
        row = df[df["is_comb"] == is_comb_value].iloc[0]
        y = row[guess_cols].values * 100  # convert to percentages
        label = "JA4+JA4S+SNI" if is_comb_value else "JA4"
        plt.plot(x, y, marker="o", label=label)

    # Formatting
    plt.xlabel("Pořadí odhadu")
    plt.ylabel("Přesnost [%]")
    plt.title("Přesnost pro každý odhad")

    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "../dias/guess_accuracy.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )


# Load and call
df = pd.read_csv("outs/ex5-mobile-filters.csv", sep=";")
df = df.tail(2)
plot_guess_accuracy(df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and filter datasets
df_iscx = pd.read_csv("outs/ex2-iscx.csv", sep=";")
df_mobile = pd.read_csv("outs/ex2-mobile.csv", sep=";")
df_iscx = df_iscx[df_iscx["is_comb"]]
df_mobile = df_mobile[df_mobile["is_comb"]]

# Plot settings
sns.set(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "legend.title_fontsize": 16,
    }
)

# --------- ISCX plot ---------
plt.figure(figsize=(6, 6))
sns.lineplot(
    data=df_iscx,
    x="min_support",
    y="avg_len_of_candidates",
    hue="items",
    marker="o",
    errorbar=None,
)
plt.gca().invert_xaxis()
plt.title("ISCX")
plt.xlabel("Minimální podpora (klesající)")
plt.ylabel("Průměrná velikost množiny")
plt.legend(title="Kombinace položek", loc="best", frameon=True)
plt.tight_layout()
plt.savefig("../dias/ex2-candidates_len-iscx.pdf", format="pdf", bbox_inches="tight")
plt.show()

# --------- MOBILE plot ---------
plt.figure(figsize=(6, 6))
sns.lineplot(
    data=df_mobile,
    x="min_support",
    y="avg_len_of_candidates",
    hue="items",
    marker="o",
    errorbar=None,
)
plt.gca().invert_xaxis()
plt.title("Mobile")
plt.xlabel("Minimální podpora (klesající)")
plt.ylabel("Průměrná velikost množiny")
plt.legend(title="Kombinace položek", loc="best", frameon=True)
plt.tight_layout()
plt.savefig("../dias/ex2-candidates_len-mobile.pdf", format="pdf", bbox_inches="tight")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("exp1.csv", sep=";")

df_not_comb = df[~df["is_comb"]]
df_comb = df[df["is_comb"]]

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_not_comb, x="items", y="accuracy_overall", errorbar="sd", marker="o"
)
plt.xticks(rotation=-90)
plt.title("Accuracy vs Number of Items")
plt.xlabel("Number of Items Used for Identification")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("../dias/accuracy_vs_items_comb.pdf")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_not_comb, x="items", y="avg_len_of_candidates", errorbar="sd")
plt.xticks(rotation=-90)
plt.title("Candidate Set Size per Parameter")
plt.xlabel("Number of Items")
plt.ylabel("Average Candidate Size")
plt.ylim(2, 2.2)
plt.tight_layout()
plt.savefig("../dias/candidate_size_comb.pdf")


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_comb, x="items", y="accuracy_overall", errorbar="sd", marker="o")
plt.xticks(rotation=-90)
plt.title("Accuracy vs Number of Items")
plt.xlabel("Number of Items Used for Identification")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("../dias/accuracy_vs_items.pdf")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_comb, x="items", y="avg_len_of_candidates", errorbar="sd")
plt.xticks(rotation=-90)
plt.title("Candidate Set Size per Parameter")
plt.xlabel("Number of Items")
plt.ylabel("Average Candidate Size")
plt.ylim(1.5, 1.7)
plt.tight_layout()
plt.savefig("../dias/candidate_size.pdf")

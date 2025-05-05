import pandas as pd

df = pd.read_csv("outs/ex1-iscx.csv", sep=";")
df = df.filter(
    items=[
        "is_comb",
        "items",
        "accuracy_overall",
        "avg_len_of_candidates",
        "time_elapsed",
    ],
    axis=1,
)

df = df.round(3)
df["items_length"] = df["items"].str.len()
df = df.sort_values(by=["items_length", "items"], ascending=[True, True]).drop(
    columns=["items_length"]
)
print(df["items"].unique())
exit()
df_not_comb = df[~df["is_comb"]]
df_comb = df[df["is_comb"]]
df_not_comb = df_not_comb.drop(columns=["is_comb"])
df_comb = df_comb.drop(columns=["is_comb"])

# Convert to LaTeX tabular format
latex_table = df_not_comb.to_latex(
    index=False,
    caption="not comb",
    label="tab:combo_accuracy",
)

# Save to .tex file
with open("../dias/tab-items-accuracy.tex", "w") as f:
    f.write(latex_table)

latex_table = df_comb.to_latex(
    index=False,
    caption="comb",
    label="tab:combo_accuracy",
)

# Save to .tex file
with open("../dias/tab-items-accuracy-comb.tex", "w") as f:
    f.write(latex_table)

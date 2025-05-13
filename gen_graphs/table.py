import pandas as pd

# Load datasets
df_mobile = pd.read_csv("outs/ex1-mobile.csv", sep=";")
df_iscx = pd.read_csv("outs/ex1-iscx.csv", sep=";")


def sort_by_length_and_value(df):
    # Sort the DataFrame by the length of the 'items' column and then by the 'items' column itself
    df["items_length"] = df["items"].str.len()
    df = df.sort_values(by=["items_length", "items"], ascending=[True, True]).drop(
        columns=["items_length"]
    )
    return df


# Define a function to process and generate a merged table with two-line headers
def process_and_save_merged_table_with_headers(df_mobile, df_iscx, ja_version, is_comb):
    # Add a column to indicate the dataset
    df_mobile["dataset"] = "mobile"
    df_iscx["dataset"] = "iscx"

    df_mobile = sort_by_length_and_value(df_mobile)
    df_iscx = sort_by_length_and_value(df_iscx)

    # Concatenate the two datasets
    df = pd.concat([df_mobile, df_iscx])

    # Filter required columns
    df = df.filter(
        items=[
            "dataset",
            "is_comb",
            "ja_version",
            "items",
            "accuracy_overall",
            "avg_len_of_candidates",
            "time_elapsed",
        ],
        axis=1,
    )

    # Filter by ja_version and is_comb
    filtered_df = df[(df["ja_version"] == ja_version) & (df["is_comb"] == is_comb)]

    # Split by dataset
    df_mobile = filtered_df[filtered_df["dataset"] == "mobile"].drop(
        columns=["dataset", "is_comb", "items", "ja_version"]
    )
    df_iscx = filtered_df[filtered_df["dataset"] == "iscx"].drop(
        columns=["dataset", "is_comb", "ja_version", "time_elapsed"]
    )

    # Rename columns to include dataset name
    df_mobile.columns = pd.MultiIndex.from_product([["mobile"], df_mobile.columns])
    df_iscx.columns = pd.MultiIndex.from_product([["iscx"], df_iscx.columns])

    # Merge the two datasets horizontally
    merged_df = pd.concat(
        [df_iscx.reset_index(drop=True), df_mobile.reset_index(drop=True)], axis=1
    )
    merged_df = merged_df.rename(
        columns={
            "accuracy_overall": "Přesnost",
            "avg_len_of_candidates": "Délka",
            "time_elapsed": "Čas",
            "items": "Kombinace položek",
        }
    )
    # Generate LaTeX table
    comb_status = "comb" if is_comb else "not_comb"
    latex_table = merged_df.to_latex(
        index=False,
        caption=f"Merged{comb_status}AccuracyJa{ja_version}",
        label=f"tab:merged-{comb_status}-accuracy-ja{ja_version}",
        multirow=True,
        multicolumn=True,
        float_format="%.3f",
    )

    # Save to .tex file
    filename = f"../tables/merged-{comb_status}-accuracy-ja{ja_version}.tex"
    with open(filename, "w") as f:
        f.write(latex_table)


# Process and save merged tables for all combinations of ja_version and is_comb
for ja_version in [3, 4]:
    for is_comb in [True, False]:
        process_and_save_merged_table_with_headers(
            df_mobile, df_iscx, ja_version, is_comb
        )

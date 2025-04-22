"""
File: aggregate.py
Description:    Simple script to aggregate and analyze data from a CSV file.
                Used only for analysis purposes.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 01/11/2024
Updated: 22/04/2025
"""

import pandas as pd
import argparse


def process_file(file_path, max_unique_values=10):
    ds = pd.read_csv(file_path, sep=";")
    columns = ds.columns

    counts = {col: ds[col].value_counts() for col in columns}
    percentages = {col: ds[col].value_counts(normalize=True) * 100 for col in columns}
    total_count = len(ds)

    for col in columns:
        unique_count = ds[col].nunique()
        column_total_count = ds[
            col
        ].count()  # Total count for the column (non-NaN values)
        uniqueness = (unique_count / total_count) * 100
        percentage_of_total = (column_total_count / total_count) * 100
        print(
            f"Column: {col} Unique values: {unique_count} ({uniqueness:.2f}%), Total count: {column_total_count} ({percentage_of_total:.2f}%)"
        )

        if unique_count > max_unique_values:
            print(f"  Showing top {max_unique_values} values:")
            top_values = counts[col].head(max_unique_values)
        else:
            top_values = counts[col]

        for idx, count in top_values.items():
            percent = percentages[col][idx]
            print(f"  {idx}: {count} ({percent:.2f}%)")

        print("\n")
    print(f"Total count (rows): {total_count}")


def process_file_to_latex(file_path, max_unique_values=10, output_file="table.tex"):
    ds = pd.read_csv(file_path, sep=";")
    columns = ds.columns

    # Prepare a summary DataFrame
    summary_data = []
    total_count = len(ds)

    for col in columns:
        unique_count = ds[col].nunique()
        column_total_count = ds[
            col
        ].count()  # Total count for the column (non-NaN values)
        uniqueness = (unique_count / total_count) * 100
        percentage_of_total = (column_total_count / total_count) * 100

        summary_data.append(
            {
                "Column": col,
                "Unique Values": unique_count,
                "Uniqueness (\%)": f"{uniqueness:.2f}",
                "Total Count": column_total_count,
                "Percentage of Total (\%)": f"{percentage_of_total:.2f}",
            }
        )

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Export to LaTeX
    with open(output_file, "w") as f:
        f.write(
            summary_df.to_latex(
                index=False, caption="Data Summary", label="tab:data_summary"
            )
        )

    print(f"LaTeX table saved to {output_file}")


def check_filename_appname_uniqueness(file_path):
    ds = pd.read_csv(file_path, sep=";")
    grouped = ds.groupby("Filename")["AppName"].nunique()
    non_unique_filenames = grouped[grouped > 1]

    if non_unique_filenames.empty:
        print("All filenames are associated with only one app name.")
    else:
        print("The following filenames are associated with more than one app name:")
        for filename, unique_app_count in non_unique_filenames.items():
            print(f"  {filename}: {unique_app_count} unique app names")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--max_unique_values", type=int, default=10)
    parser.add_argument("--latex", type=str, help="Export to LaTeX format")

    args = parser.parse_args()
    process_file_to_latex(args.file_path, args.max_unique_values, args.latex)
    check_filename_appname_uniqueness(args.file_path)

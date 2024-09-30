import pandas as pd
import argparse

def process_file(file_path, max_unique_values=10):
    ds = pd.read_csv(file_path, sep=';')
    columns = ds.columns

    counts = {col: ds[col].value_counts() for col in columns}
    percentages = {col: ds[col].value_counts(normalize=True) * 100 for col in columns}
    total_count = len(ds)

    for col in columns:
        unique_count = ds[col].nunique()
        uniqueness= (unique_count / total_count) * 100
        print(f"Column: {col} (Unique values: {unique_count}, Uniqueness: {uniqueness:.2f}%)")
        
        if unique_count > max_unique_values:
            print(f"  Showing top {max_unique_values} values:")
            top_values = counts[col].head(max_unique_values)
        else:
            top_values = counts[col]
        
        for idx, count in top_values.items():
            percent = percentages[col][idx]
            print(f"  {idx}: {count} ({percent:.2f}%)")
        
        print("\n")
    print(f"Total count: {total_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('--max_unique_values', type=int, default=10)

    args = parser.parse_args()
    process_file(args.file_path, args.max_unique_values)

from prefixspan import PrefixSpan
import csv
import argparse

def parse_csv(csv_file):
    db = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if i == 0: 
                continue
            db.append([item for item in row])
    return db

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="Path to csv file")
    args = parser.parse_args()

    db = parse_csv(args.file_path)

    ps = PrefixSpan(db)
    patterns = ps.frequent(2)
    print(patterns)

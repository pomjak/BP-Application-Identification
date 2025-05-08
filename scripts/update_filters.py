import os
import ast

CONFIG_PATH = "config.py"
# Define combinations of filters
FILTER_COMBINATIONS = [
    [
        {"operator": "==", "length": 1, "head": 2},
    ],
    [
        {"operator": "==", "length": 3, "head": 2},
    ],
    [
        {"operator": "==", "length": 2, "head": 2},
    ],
    [
        {"operator": "==", "length": 1, "head": 2},
        {"operator": "==", "length": 2, "head": 2},
    ],
    [
        {"operator": "==", "length": 1, "head": 2},
        {"operator": "==", "length": 3, "head": 2},
    ],
    [
        {"operator": "==", "length": 2, "head": 2},
        {"operator": "==", "length": 3, "head": 2},
    ],
    [
        {"operator": "==", "length": 1, "head": 2},
        {"operator": "==", "length": 2, "head": 2},
        {"operator": "==", "length": 3, "head": 2},
    ],
    # Add more combinations here
]

INDEX_TRACK_FILE = ".filter_index"


def get_next_filter_index():
    """Read the index from the file or initialize it if it doesn't exist."""
    if not os.path.exists(INDEX_TRACK_FILE):
        with open(INDEX_TRACK_FILE, "w") as f:
            f.write("0")
        return 0
    with open(INDEX_TRACK_FILE, "r") as f:
        index = int(f.read().strip())
    return index


def save_next_filter_index(index):
    """Save the next filter index to track progress."""
    with open(INDEX_TRACK_FILE, "w") as f:
        f.write(str(index))


def update_config(filter_combination):
    """Update the config file with a new combination of filters."""
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()

    new_lines = []
    in_filter = False
    for line in lines:
        if line.strip().startswith("PATTERN_FILTERS"):
            in_filter = True
            new_lines.append("PATTERN_FILTERS = [\n")
            for filter_item in filter_combination:
                new_lines.append(f"    {filter_item},\n")
            new_lines.append("]\n")
        elif in_filter and line.strip().startswith("]"):
            in_filter = False
            continue  # skip
        elif not in_filter:
            new_lines.append(line)

    with open(CONFIG_PATH, "w") as f:
        f.writelines(new_lines)


def main():
    """Main function to run the experiment with a filter combination."""
    index = get_next_filter_index()

    # If we have applied all combinations, stop
    if index >= len(FILTER_COMBINATIONS):
        print("All filter combinations have been used.")
        return

    # Get the filter combination to apply
    filter_combination = FILTER_COMBINATIONS[index]
    update_config(filter_combination)
    print(
        f"Applied combination {index + 1}/{len(FILTER_COMBINATIONS)}: {filter_combination}"
    )
    save_next_filter_index(index + 1)


if __name__ == "__main__":
    main()

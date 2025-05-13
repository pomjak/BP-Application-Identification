# Identify 

_Identify_ is program that implements __identification__ of applications using _JA3_ or _JA4_ fingerprints. Then proceeds to __eliminate__ possible candidates using __context-aware detection__. See technical report for more information.

### Author
- **Name:** Pomsar Jakub  
- **Xlogin:** `xpomsa00`  
- **Created:** 21/04/2025

## Getting Started
This guide walks you through installing and running the `Identify` program step-by-step. Also provides detailed project structure. 

### Structure
```
.
├── aggregate.py    (simple script for aggregating data from datasets)
├── config.py       (configuration file for filters and itemsets)
├── out             (folder containing all outputs of experiments)
├── data
│   ├── gh-repos.md (link for used datasets)
│   ├── iscx.csv    
│   ├── iscx-raw.csv
│   └── mobile_desktop_apps_raw.csv
├── identify        (folder containing business logic)
│   ├── command_line_parser.py
│   ├── database.py
│   ├── fingerprinting.py
│   ├── __init__.py
│   ├── ja_context.py
│   ├── logger.py
│   └── pattern_matching.py
├── out             (folder containing all outputs of experiments)
├── main.py 
├── Makefile        (Makefile for simpler usage)
├── README.md
└── requirements.txt (needed requirements for successful execution) 

```
Prerequisites
* Python 3.10+
* `make`


### Installation

The program is set up to use a virtual environment. You can create the environment using the `make` or `make env` command, and then activate it with:

```bash
source env/bin/activate
```

#### Manual Installation

If you prefer to set up the environment manually:

1. Create the virtual environment:
   ```bash
   python3 -m venv env
   ```
2. Install the required packages:
   ```bash
   env/bin/pip install -r requirements.txt
   ```
3. Activate the environment:
   ```bash
   source env/bin/activate
   ```


## Usage

This section provides usage instructions for the program and the Makefile.

### Manual execution

To run the program manually, following command can be used:
```bash
python3 main.py -d <dataset> -f <3|4> -w <window> -m <min_support> -c <candidates>
```
Where:
- `-d <dataset>`: Path to the dataset (e.g., `data/iscx.csv`)
- `-f <3|4>`: Fingerprint version (`3` for JA3, `4` for JA4) (this specifies what version of method is used to generate base candidate set)
- `-w <window>`: Sliding window width (integer)
- `-m <min_support>`: Minimum support for Apriori (float, 0.0–1.0)
- `-c <candidates>`: Number of candidates to identify (integer)
- `-h`, `--help`: Show help message

Logs are appended by default. It's recommended to delete the log file before each manual run for more clarity between runs.

### Result Sections Description

The program outputs four distinct sections:
If the `-f` is set to **4** (JA4 fingerprinting), then:
1. **Fingerprint (JA4)**:
   - Results based on the pure **JA4** fingerprinting method.

2. **Fingerprint Combination (JA4 + JA4S + SNI)**:
   - Results based on the combination of the chosen fingerprint method (JA4 or JA3) along with **JA4S** and **SNI**.

3. **Fingerprint with Context (JA4)**:
   - Results using **JA4** as the base method, but with context applied (items selected in `config.py`).

4. **Fingerprint Combination with Context (JA4 + JA4S + SNI)**:
   - Results using **JA4 + JA4S + SNI** as the base method, with context applied (as selected in `config.py`).





## Experiments
This section provides instructions for running the experiments executed in technical report. Since all experiments were conducted as iterations over different parameters (available for review in the outs/ folder as CSV files), it is recommended to run the program with only a few selected parameters. Running multiple configurations without filters and with a low minimum support may take a considerable amount of time. 

Before running the experiments, make sure to create virtual environment, install the required packages and **activate the environment** as described in the [Installation](#installation) section.

### Experiment 1 (Selecting Ideal Itemsets)

The following command can be used to run the first experiment. By default, the configuration is set to use the items **JA4**, **JA4S**, and **SNI**. You can modify this selection in the `config.py` file, specifically in the `columns_to_keep_for_context` function.

The following command runs the program for the **JA4** and **JA4+JA4S+SNI** methods (these methods are used to generate the base candidate set, which is then shortened using context identification; frequent itemsets will be found based on whatever items are selected in `config.py`):

```bash
python main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3
```

For the **JA3** and **JA3+JA3S+SNI** methods (generating the base candidate set):

```bash
python main.py -d data/mobile_desktop_apps_raw.csv -f 3 -w 25 -m 0.25 -c 3
```

To edit the configuration, you can modify the `config.py` file.

In the file, locate the section for context identification and modify the attributes as needed. For example:

```python
#! INSERT HERE ATTRIBUTES FOR CONTEXT IDENTIFICATION !#

columns_to_keep_for_context = [
    # JA3,
    # JA3_S,
    JA4,  # Mostly used items in experiments are JA4, JA4S, SNI
    JA4_S,
    SNI,
    # Add other items as necessary
]
```

You can add or remove items based on the experiment you want to run. The items listed here are used mostly used throughout the experiments.

### Experiment 2 (Selecting min_support)

This experiment uses only version 4 of fingerprinting, specifically **JA4** and **JA4+JA4S+SNI**. The key parameter in this experiment is the minimum support, denoted as `-m`. By default, the configuration is set to use the items **JA4**, **JA4S**, and **SNI**. You can modify this selection in the `config.py` file, specifically in the `columns_to_keep_for_context` function.

To run the experiment, use the following command:

```bash
python main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3
```

### Experiment 2.1 (Selecting Filters)

This experiment uses only version 4 of fingerprinting, specifically **JA4** and **JA4+JA4S+SNI**. The key goal of the experiment is to find the best match for methods using different filters. Filters can be enabled in the `config.py` file within the `PATTERN_FILTERS` list.

By default, filters are disabled, but they can be enabled by uncommenting the relevant lines in `config.py`. To run the experiment without filters, use the following command:

```bash
python main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3
```

To enable filters, uncomment the relevant filter selections in `config.py`. Here’s an example:

```python
PATTERN_FILTERS = [
    # iscx ja4
    # {"operator": "==", "length": 1, "head": 5},
    # {"operator": "==", "length": 3, "head": 10},
    # iscx comb
    # {"operator": "==", "length": 3, "head": 5},
    # mobile ja4
    # {"operator": "==", "length": 2, "head": 5},
    # {"operator": "==", "length": 3, "head": 10},
    # mobile comb
    # {"operator": "==", "length": 1, "head": 2},
    # {"operator": "==", "length": 3, "head": 2},
]
```

For example, if you are using the **ISCX** dataset with the **JA4** base method, you should uncomment the 3rd and 4th lines to enable the corresponding filters:

```python
PATTERN_FILTERS = [
    # iscx ja4
    {"operator": "==", "length": 1, "head": 5},
    {"operator": "==", "length": 3, "head": 10},
    # iscx comb
    # {"operator": "==", "length": 3, "head": 5},
    # mobile ja4
    # {"operator": "==", "length": 2, "head": 5},
    # {"operator": "==", "length": 3, "head": 10},
    # mobile comb
    # {"operator": "==", "length": 1, "head": 2},
    # {"operator": "==", "length": 3, "head": 2},
]
```

Be sure to select only the appropriate filter for your dataset and method. Expect the results to change depending on the selected filter.

### Experiment 3 (Window Size Variation)

In this experiment, the base methods chosen are the same as in the previous experiments, and the filters in use are also commented out in `config.py`. The key parameter in this experiment is the **window size**, denoted as `-w`. 

To run the experiment with a window size of **15**, use the following command:

```bash
python main.py -d data/iscx.csv -f 4 -w 15 -m 0.01 -c 3
```

Alternatively, to run the experiment with a window size of **3**, use this command:

```bash
python main.py -d data/iscx.csv -f 4 -w 3 -m 0.01 -c 3
```

### Experiment 4 (Candidate Set Size)

In this experiment, the maximum number of candidates is set by the `-c` parameter. The best results were achieved with a minimum support of **0.01**, using filters (edit in `config.py`), a selected window size of **3**, and itemsets **JA4**, **JA4S**, and **SNI** (edit in `config.py`).

To run the experiment with the **ISCX** dataset, use the following command:

```bash
python main.py -d data/iscx.csv -f 4 -w 3 -m 0.01 -c 4
```

To run the experiment with the **Mobile Desktop Apps** dataset, use this command:

```bash
python main.py -d data/mobile_desktop_apps_raw.csv -f 4 -w 3 -m 0.01 -c 9
```


### Acknowledgments

This README was edited with the assistance of ChatGPT for grammar and stylistic improvements.

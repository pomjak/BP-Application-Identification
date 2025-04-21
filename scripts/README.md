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
├── config.py       (configuration file specifying column names of datasets and DEBUG option)
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
The simplest way to run all experiment mentioned in technical report is :
```
make 
``` 
This sets up virtual environment, in which installs all needed requirements and executes program with preset parameters for experiments.

#### Manual Installation
If manual installation is preferred, then:
```
make env
```
or
```
python3 -m venv env
```
to create environment. 
Following installing requirements:
```
make install
```
or
```
env/bin/pip install -r requirements.txt
```

## Usage
This usage provides 
### Using Makefile
To run all experiments (as mentioned earlier):
```
make 
```
or
```
make experiments
```
All outputs and logs of experiments will be __overwritten__ in folder `out/`

To run just specific experiment N where N represents number of experiment:
```
make expN 
```
Output of program (expN.out) and log(expN.log) is __saved__ in folder `out\`

To clean generated files and `pycache`:
```
make clean
```
or
```
make clean-all
```
to also clean environment folder.
Also help is available:
```
make help
```

### Manual execution
To run the program manually, following command can be used:
```
python3 main.py -d <dataset> -f <3|4> -w <window> -m <min_support> -c <candidates>
```
Where:
- `-d <dataset>`: Path to the dataset (e.g., `data/iscx.csv`)
- `-f <3|4>`: Fingerprint version (`3` for JA3, `4` for JA4)
- `-w <window>`: Sliding window width (integer)
- `-m <min_support>`: Minimum support for Apriori (float, 0.0–1.0)
- `-c <candidates>`: Number of candidates to identify (integer)
- `-h`, `--help`: Show help message

Logs are appended by default. It's recommended to delete the log file before each manual run.
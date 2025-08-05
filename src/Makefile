# File: Makefile
# Description: Makefile for managing the project
# Author: Pomsar Jakub
# Xlogin: xpomsa00
# Created: 21/04/2025
# Updated: 21/04/2025

# Path to the virtual environment directory
VENV_DIR = .env
# Path to the Python executable in the virtual environment
VENV_PYTHON = $(VENV_DIR)/bin/python
# Path to the pip executable in the virtual environment
VENV_PIP = $(VENV_DIR)/bin/pip
# Path to the requirements file
REQS_FILE = requirements.txt
# Name of the main Python program
PROGRAM = main.py

.PHONY: install env clean clean-all reqs help

install: env requirements.txt
	@echo "Installing dependencies..."
	$(VENV_PIP) install -r $(REQS_FILE)  

env:
	@test -d $(VENV_DIR) || (echo "Creating virtual environment..." && python3 -m venv $(VENV_DIR))

ex1:
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3
	$(VENV_PYTHON) main.py -d data/mobile_desktop_apps_raw.csv -f 3 -w 25 -m 0.25 -c 3

ex2:
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3

ex2_1:
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 15 -m 0.25 -c 3

ex3:
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 15 -m 0.01 -c 3
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 3 -m 0.01 -c 3

ex4:
	$(VENV_PYTHON) main.py -d data/iscx.csv -f 4 -w 3 -m 0.01 -c 4
	$(VENV_PYTHON) main.py -d data/mobile_desktop_apps_raw.csv -f 4 -w 3 -m 0.01 -c 9

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ identify/__pycache__ identify.log  $(OUT_FOLDER)

clean-all: clean
	@echo "Cleaning up virtual environment..."
	rm -rf $(VENV_DIR)

reqs:
	@echo "Updating requirements.txt..."
	pipreqs . --force

help:
	@echo "Makefile commands:"
	@echo "  install     - Install dependencies from $(REQS_FILE)"
	@echo "  reqs        - Update $(REQS_FILE) with current dependencies"
	@echo "  ex1         - Run Experiment 1: Selecting Ideal Itemsets"
	@echo "  ex2         - Run Experiment 2: Selecting min_support"
	@echo "  ex2_1       - Run Experiment 2.1: Selecting Filters"
	@echo "  ex3         - Run Experiment 3: Window Size Variation"
	@echo "  ex4         - Run Experiment 4: Candidate Set Size"
	@echo "  env         - Create a virtual environment"
	@echo "  clean-all   - Clean all temporary files and logs, including the virtual environment"
	@echo "  clean       - Clean temporary files and logs"
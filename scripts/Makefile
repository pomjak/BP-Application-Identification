# File: Makefile
# Description: Makefile for managing the project
# Author: Pomsar Jakub
# Xlogin: xpomsa00
# Created: 21/04/2025
# Updated: 21/04/2025

# Path to the virtual environment directory
VENV_DIR = .env
# Path to the Python executable in the virtual environment
PYTHON = $(VENV_DIR)/bin/python
# Path to the pip executable in the virtual environment
PIP = $(VENV_DIR)/bin/pip
# Path to the requirements file
REQS_FILE = requirements.txt
# Path to the outputs directory
OUT_FOLDER = out
# Name of the main Python program
PROGRAM = main.py

.PHONY: install env experiments clean clean-all reqs help

experiments: exp1 exp2
	@echo "All experiments completed. Check the output files in $(OUT_FOLDER)."

exp1:
	@echo "Running experiment 1..."
	@mkdir -p $(OUT_FOLDER)
	$(PYTHON) $(PROGRAM) -d data/iscx.csv -w 25 -m 0.01 -c 4 > $(OUT_FOLDER)/exp1.out
	@echo "Experiment 1 completed. Output saved to exp1.out"

exp2: 
	@echo "Running experiment 2..."
	@mkdir -p $(OUT_FOLDER)
	$(PYTHON) $(PROGRAM) -d data/iscx.csv -w 25 -m 0.1 -c 4 > $(OUT_FOLDER)/exp2.out
	@echo "Experiment 2 completed. Output saved to exp2.out"

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
	@echo "  run         - Run the $(PROGRAM) script"
	@echo "  reqs        - Update $(REQS_FILE) with current dependencies"
	@echo "  experiments - Run experiments"
	@echo "  clean       - Clean temporary files and logs"
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
	@echo "  experiments - Run all experiments"
	@echo "  expN        - Run experiment N"
	@echo "  env         - Create a virtual environment"
	@echo "  clean-all   - Clean all temporary files and logs, including the virtual environment"
	@echo "  clean       - Clean temporary files and logs"
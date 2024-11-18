"""
File: logger.py
Description: This file contains Logger class for creating log information.
Author: Pomsar Jakub
Xlogin: xpomsa00
Created: 15/11/2024
Updated: 18/11/2024
"""

import inspect
from datetime import datetime
import os


class Logger:
    DEFAULT_LOG_FILE = "identify.log"

    def __init__(self, log_file=None):
        self.log_file = log_file or self.DEFAULT_LOG_FILE

    def __enter__(self):
        self.log = open(self.log_file, "a")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.close()

    def _log(self, level, message):
        frame = inspect.currentframe().f_back.f_back
        full_path = frame.f_code.co_filename
        project_folder = os.path.basename(os.path.dirname(full_path))
        filename = os.path.basename(full_path)
        lineno = frame.f_lineno
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} [{level}] {project_folder}/{filename}:{lineno} - {message}\n"

        self.log.write(log_message)
        self.log.flush()

    def debug(self, message):
        self._log("DEBUG", message)

    def info(self, message):
        self._log("INFO", message)

    def warn(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

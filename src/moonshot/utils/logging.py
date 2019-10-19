"""Utility functions for logging.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime
import logging as base_logging


from absl.app import logging


def absl_file_logger(path, name="log"):
    """Add a file handler to the abseil logging system."""
    logger = logging.get_absl_logger()


    uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(path, f"{name}.{uid}")

    file_handler = base_logging.FileHandler(log_file)
    file_handler.setFormatter(
        base_logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)

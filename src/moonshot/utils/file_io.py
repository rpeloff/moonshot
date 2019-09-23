"""Utility functions for file input/output.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os


from absl.app import logging


def check_create_dir(path):
    """Check if a directory exists otherwise create it."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_csv(path, *columns, column_names=None):
    """Write a simple csv file with comma delimiter."""
    check_create_dir(path)

    with open(path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")

        if column_names is not None:
            csv_writer.writerow(column_names)

        csv_writer.writerows(zip(*columns))


def read_csv(path, skip_first=False, delimiter=","):
    """Read a csv file."""
    logging.log(logging.INFO, "Reading csv file: {}".format(path))
    with open(path, "r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        rows = []
        for idx, row in enumerate(csv_reader):
            if skip_first and idx == 0:
                logging.log(
                    logging.INFO, "Skipping first row: {}".format(row))
            else:
                rows.append(row)
        csv_data = tuple(zip(*rows))

    return csv_data

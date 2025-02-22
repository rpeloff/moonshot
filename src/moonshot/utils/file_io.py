"""Utility functions for file input/output.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import json
import os


from absl.app import logging


def check_create_dir(path):
    """Check if a directory exists otherwise create it."""
    if os.path.splitext(path)[1] == "":  # assume basename is directory not file
        dir_name = path
    else:
        dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_csv(path, *columns, column_names=None):
    """Write a simple csv file with comma delimiter."""
    check_create_dir(path)
    logging.log(logging.INFO, "Writing csv file: {}".format(path))

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


def write_json(path, json_dict, indent=4):
    """Write a dictionary to a json file."""
    check_create_dir(path)
    logging.log(logging.INFO, "Writing json file: {}".format(path))

    with open(path, "w") as json_file:
        json.dump(json_dict, json_file, indent=indent)


def read_json(path):
    """Read a json file."""
    logging.log(logging.INFO, "Reading json file: {}".format(path))

    with open(path, "r") as json_file:
        json_dict = json.load(json_file)

    return json_dict

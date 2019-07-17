"""
Load Flickr 8k splits from Flicker 8k text caption corpus.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""


import os


def get_flickr8k_train_test_dev(text_base_dir):
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        if subset not in set_dict:
            set_dict[subset] = []
        subset_fn = os.path.join(text_base_dir, "Flickr_8k.{}Images.txt".format(subset))
        print("Reading:", subset_fn)
        with open(subset_fn) as f:
            for line in f:
                set_dict[subset].append(os.path.splitext(line.strip())[0])
    return set_dict

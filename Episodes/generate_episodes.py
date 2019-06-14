#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from os import path
import argparse
import os
import datetime
import numpy as np
import sys
import time
import few_shot_learning_library
from scipy.spatial.distance import cdist
import hashlib
import few_shot_learning_library
import re
import filecmp

sys.path.append("..")
from paths import data_path
from paths import feats_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import speech_model_library
import vision_model_library

data_path = path.join("..", data_path)
feats_path = path.join("..", feats_path)

import random

default_model_lib = {
    
        "speech_data_type": "TIDigits",
        "features_type": "mfcc", 
        "data_tag": "gt",
        "speech_subset": "test",        
        "max_frames": None,
        "speech_input_dim": None,
        "image_data_type": "MNIST",
        "image_input_dim": None,
        "image_subset": "test"

    }

def arguments_for_library_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_data_type", type=str, choices=["train", "val", "test"], default=default_model_lib["speech_data_type"])
    parser.add_argument("--features_type", type=str, choices=["train", "val", "test"], default=default_model_lib["features_type"])
    parser.add_argument("--data_tag", type=str, choices=["train", "val", "test"], default=default_model_lib["data_tag"])
    parser.add_argument("--speech_subset", type=str, choices=["train", "val", "test"], default=default_model_lib["speech_subset"])
    parser.add_argument("--max_frames", type=int, default=default_model_lib["max_frames"])
    parser.add_argument("--image_data_type", type=str, choices=["train", "val", "test"], default=default_model_lib["image_data_type"])
    parser.add_argument("--image_subset", type=str, choices=["train", "val", "test"], default=default_model_lib["image_subset"])
    parser.add_argument("--M", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--Q", type=int)    
    return parser.parse_args()

def library_setup():

    parameters = arguments_for_library_setup()

    model_lib = default_model_lib.copy()

    model_lib["speech_data_type"] = parameters.speech_data_type
    model_lib["features_type"] = parameters.features_type
    model_lib["data_tag"] = parameters.data_tag
    model_lib["speech_subset"] = parameters.speech_subset
    model_lib["max_frames"] = parameters.max_frames
    model_lib["image_data_type"] = parameters.image_data_type
    model_lib["image_subset"] = parameters.image_subset
    model_lib["M"] = parameters.M
    model_lib["K"] = parameters.K
    model_lib["Q"] = parameters.Q

    model_lib["speech_data_dir"] = path.join(
    	feats_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"], 
    	model_lib["data_tag"] + "_" + model_lib["speech_subset"] + "_" + model_lib["features_type"] + ".npz"
    	)
    model_lib["image_data_dir"] = path.join(
    	feats_path, model_lib["image_data_type"], model_lib["image_subset"] + ".npz"
    	)

    model_lib["speech_input_dim"] = 13
    model_lib["max_frames"] = 100
    model_lib["image_input_dim"] = 28*28

    model_lib["name"] = "M_{}_K_{}_Q_{}_{}_{}_{}_{}".format(
    		model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["speech_data_type"],
    		model_lib['speech_subset'],model_lib["image_data_type"], model_lib["image_subset"]
    		)

    base_dir = path.join(".", "Episode_files")
    util_library.check_dir(base_dir)
    model_lib["data_fn"] = path.join(base_dir, model_lib["name"])

    return model_lib

def generate_image_lists(images, labels, file_fn=None):
	
	arrays = []
	keys = []
	lab = []
	if file_fn is not None: file = open(file_fn, "w")
	for i in range(images.shape[0]):
		
		arrays.append(images[i:i+1, :])
		hasher = hashlib.md5(repr(images[i:i+1, :]).encode("ascii"))
		key = hasher.hexdigest()
		keys.append(key)
		lab.append(str(np.argmax(labels[i:i+1, :], axis=1)[0]))

		if file_fn is not None: file.write("{} {} {}\n".format(i, lab[i], keys[i]))

	if file_fn is not None: file.close()
	return arrays, keys, lab

def find_data_of_key(query_keys, data, keys, labels, lengths=None):

	query_dict = {}
	query_dict["data"] = []
	query_dict["keys"] = []

	if lengths is not None: query_dict["lengths"] = []

	for i in range(len(query_keys)):

		index = np.where(np.asarray(keys) == query_keys[i])[0][0]	
		if i == 0: label = labels[index]
		elif labels[index] != label: 
			print("Labels don't match!")
			continue
			
		query_dict["data"].append(data[index])
		query_dict["keys"].append(keys[index])
		if lengths is not None: query_dict["lengths"].append(lengths[index])


	if lengths is not None: return query_dict["data"], query_dict["keys"], query_dict["lengths"]
	return query_dict["data"], query_dict["keys"]

def read_in_episodes(list_fn):

	index_dict = {}
	curr_episode = ""
	episodes = {}
	currently_reading_in = ""
	currently_reading_in_section = "" 

	for i, line in enumerate(open(list_fn, 'r')):
		if re.search("Episode", line): 
			currently_reading_in = ""
			episode_num = line.strip().split(" ")[-1]
			curr_episode = str(episode_num)
			episodes[curr_episode] = {}

		elif re.search("Support set:", line): 
			currently_reading_in = "support_set"
			episodes[curr_episode]["support_set"] = {}
			episodes[curr_episode]["support_set"]["labels"] = []
			episodes[curr_episode]["support_set"]["image_keys"] = []
			episodes[curr_episode]["support_set"]["speech_keys"] = []
		elif re.search("Query:", line): 
			currently_reading_in = "query"
			episodes[curr_episode]["query"] = {}
			episodes[curr_episode]["query"]["labels"] = []
			episodes[curr_episode]["query"]["speech_keys"] = []			
		elif re.search("Matching set:", line): 
			currently_reading_in = "matching_set"
			episodes[curr_episode]["matching_set"] = {}
			episodes[curr_episode]["matching_set"]["labels"] = []
			episodes[curr_episode]["matching_set"]["image_keys"] = []

		elif re.search("Labels:", line):
			currently_reading_in_section = "labels"
		elif re.search("Image keys:", line):
			currently_reading_in_section = "image_keys"
		elif re.search("Speech keys:", line):
			currently_reading_in_section = "speech_keys"
		elif re.search("Keys:", line):
			currently_reading_in_section = "keys"
		elif len(line.strip().split()) == 0:
			currently_reading_in = ""
			currently_reading_in_section = ""
			continue

		elif currently_reading_in == "support_set" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["support_set"]["labels"].append(line_parts[0])
		elif currently_reading_in == "support_set" and currently_reading_in_section == "image_keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["support_set"]["image_keys"].extend(key)
		elif currently_reading_in == "support_set" and currently_reading_in_section == "speech_keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["support_set"]["speech_keys"].extend(key)

		elif currently_reading_in == "query" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["query"]["labels"].append(line_parts[0])
		elif currently_reading_in == "query" and currently_reading_in_section == "keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["query"]["speech_keys"].extend(key)

		elif currently_reading_in == "matching_set" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["matching_set"]["labels"].extend(line_parts[0])
		elif currently_reading_in == "matching_set" and currently_reading_in_section == "keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["matching_set"]["image_keys"].extend(key)

	return episodes

def episode_check(support_set, query, matching_set):
	check = False
	total = 0
	correct = 0
	im_keys = []
	speech_keys = []
	support_set_image_keys = []
	support_set_speech_keys = []
	query_speech_keys = []
	matching_set_image_keys = []

	for key in support_set:
		support_set_speech_keys.extend(support_set[key]["speech_keys"])

		support_set_image_keys.extend(support_set[key]["image_keys"])

	for key in query:
		query_speech_keys.extend(query[key]["keys"])

	for key in matching_set:
		matching_set_image_keys.extend(matching_set[key]["keys"])

	for support_key in support_set_image_keys:
		if support_key not in matching_set_image_keys:
			correct += 1
		else:
			im_keys.append(support_key)
		total += 1

	for support_key in support_set_speech_keys:
		if support_key not in query_speech_keys: 
			correct += 1
		else:
			im_keys.append(support_key)
		total += 1

	if correct != total: 
		check = True
		print("Im_keys: {}".format(im_keys))
		print("Sp_keys: {}".format(speech_keys))
	return check

def main():

	lib = library_setup()

	num_episodes = 400
	K = lib["K"]
	M = lib["M"]
	Q = lib["Q"]

	images, image_labels, image_keys = (
		data_library.load_image_data_from_npz(lib["image_data_dir"])
		)
	im_x, im_labels, im_keys  = images, image_labels, image_keys

	# Speech data
	sp_x, sp_labels, sp_lengths, sp_keys = (
		data_library.load_speech_data_from_npz(lib["speech_data_dir"])
		)
	max_frames = 100
	d_frame = 13
	print("\nLimiting dimensionality: {}".format(d_frame))
	print("Limiting number of frames: {}\n".format(max_frames))
	data_library.truncate_data_dim(sp_x, sp_lengths, d_frame, max_frames)

	speech_x, speech_labels, speech_lengths, speech_keys = sp_x, sp_labels, sp_lengths, sp_keys

	episodes_fn = lib["data_fn"] + "_episodes.txt"
	print("Generating epsiodes...\n")
	
	file = open(episodes_fn, "w")
	
	for episode_counter in range(1, num_episodes+1):
	
		support_set = few_shot_learning_library.construct_few_shot_support_set_with_keys(
			speech_x, speech_labels, speech_keys, speech_lengths, images, image_labels, image_keys, M, K
			)

		im_keys_to_not_include = []
		sp_keys_to_not_include = []
		for key in support_set:
			for im_key in support_set[key]["image_keys"]: im_keys_to_not_include.append(im_key)
			for sp_key in support_set[key]["speech_keys"]: sp_keys_to_not_include.append(sp_key)
		
		query_dict = few_shot_learning_library.sample_multiple_keys(
			speech_x, speech_labels, speech_keys, speech_lengths, Q, exclude_key_list=sp_keys_to_not_include
			)	

		matching_dict = few_shot_learning_library.sample_multiple_keys(
			images, image_labels, image_keys, num_to_sample=M-1, num_of_each_sample=K, exclude_key_list=im_keys_to_not_include
			)


		file.write("Episode {}\n".format(episode_counter))

		file.write("{}\n".format("Support set:"))
		file.write("{}\n".format("Labels: "))
		for key in support_set:
			file.write("{}\n".format(key))
		file.write("{}\n".format("Image keys: "))
		for key in support_set:
			for i in range(len(support_set[key]["image_keys"])):
				file.write("{}".format(support_set[key]["image_keys"][i]))
				if i == len(support_set[key]["image_keys"]) - 1: file.write("\n")
				else: file.write(" ")
		file.write("{}\n".format("Speech keys: "))
		for key in support_set:
			for i in range(len(support_set[key]["speech_keys"])):
				file.write("{}".format(support_set[key]["speech_keys"][i]))
				if i == len(support_set[key]["speech_keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("{}\n".format("Query:"))
		file.write("{}\n".format("Labels: "))
		for key in query_dict:
			file.write("{}\n".format(key))
		file.write("{}\n".format("Keys: "))
		for key in query_dict:
			for i in range(len(query_dict[key]["keys"])):
				file.write("{}".format(query_dict[key]["keys"][i]))
				if i == len(query_dict[key]["keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("{}\n".format("Matching set:"))
		file.write("{}\n".format("Labels: "))
		for key in matching_dict:
			file.write("{}\n".format(key))
		file.write("{}\n".format("Keys: "))
		for key in matching_dict:
			for i in range(len(matching_dict[key]["keys"])):
				file.write("{}".format(matching_dict[key]["keys"][i]))
				if i == len(matching_dict[key]["keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("\n")

		if episode_check(support_set, query_dict, matching_dict):
			break

	file.close()

	print("Wrote epsiodes to {}".format(episodes_fn))

	# episode_dict = read_in_episodes(episodes_fn)

	# test_fn = lib["data_fn"] + "_testing.txt"
	# file = open(test_fn, 'w')
	# for i in range(len(episode_dict)):
	# 	episode = str(i+1)
	# 	curr_episode = episode_dict[episode]
	# 	support_set = curr_episode["support_set"]
	# 	query_dict = curr_episode["query"]
	# 	matching_dict = curr_episode["matching_set"]
		
	# 	file.write("Episode {}\n".format(episode))

	# 	file.write("{}\n".format("Support set:"))

	# 	file.write("{}\n".format("Labels: "))
	# 	for i in range(len(support_set["labels"])):
	# 		file.write("{}\n".format(support_set["labels"][i]))

	# 	file.write("{}\n".format("Image keys: "))

	# 	for i in range(len(support_set["image_keys"])):
	# 		file.write("{}".format(support_set["image_keys"][i]))
	# 		if i%K == K-1: file.write("\n")
	# 		else: 
	# 			if i != len(support_set["image_keys"]) - 1: file.write(" ")
	# 			else: file.write("\n")

	# 	file.write("{}\n".format("Speech keys: "))
	# 	for i in range(len(support_set["speech_keys"])):
	# 		file.write("{}".format(support_set["speech_keys"][i]))
	# 		if i%K == K-1: file.write("\n")
	# 		else: 
	# 			if i != len(support_set["speech_keys"]) - 1: file.write(" ")
	# 			else: file.write("\n")

	# 	file.write("{}\n".format("Query:"))

	# 	file.write("{}\n".format("Labels: "))
	# 	for i in range(len(query_dict["labels"])):
	# 		file.write("{}\n".format(query_dict["labels"][i]))

	# 	file.write("{}\n".format("Keys: "))
	# 	for i in range(len(query_dict["speech_keys"])):
	# 		file.write("{}\n".format(query_dict["speech_keys"][i]))

	# 	file.write("{}\n".format("Matching set:"))

	# 	file.write("{}\n".format("Labels: "))
	# 	for i in range(len(matching_dict["labels"])):
	# 		file.write("{}\n".format(matching_dict["labels"][i]))

	# 	file.write("{}\n".format("Keys: "))
	# 	for i in range(len(matching_dict["image_keys"])):
	# 		file.write("{}".format(matching_dict["image_keys"][i]))
	# 		if i%K == K-1: file.write("\n")
	# 		else: 
	# 			if i != len(matching_dict["image_keys"]) - 1: file.write(" ")
	# 			else: file.write("\n")

	# 	file.write("\n")

	# file.close()

	# print(filecmp.cmp(test_fn, test_fn))
	# print(filecmp.cmp(episodes_fn, episodes_fn))
	# print(filecmp.cmp(episodes_fn, test_fn))

if __name__ == "__main__":
	main()

# def read_in_unimodal_speech_episodes(list_fn, in_data, in_labels, in_lengths, in_keys):

# 	index_dict = {}
# 	curr_episode = ""
# 	episodes = {}
# 	currently_reading_in = ""

# 	for i, line in enumerate(open(list_fn, 'r')):
# 		if re.search("Episode", line): 
# 			currently_reading_in = ""
# 			episode_num = line.strip().split(" ")[-1]
# 			curr_episode = str(episode_num)
# 			episodes[curr_episode] = {}

# 		elif re.search("Support set:", line): 
# 			currently_reading_in = "support_set"
# 			episodes[curr_episode]["support_set"] = {}
# 			episodes[curr_episode]["support_set"]["labels"] = []
# 			episodes[curr_episode]["support_set"]["keys"] = []
# 			episodes[curr_episode]["support_set"]["data"] = []
# 			episodes[curr_episode]["support_set"]["lengths"] = []
# 		elif re.search("Query:", line): 
# 			currently_reading_in = "query"
# 			episodes[curr_episode]["query"] = {}
# 			episodes[curr_episode]["query"]["labels"] = []
# 			episodes[curr_episode]["query"]["keys"] = []
# 			episodes[curr_episode]["query"]["data"] = []
# 			episodes[curr_episode]["query"]["lengths"] = []
			
# 		elif re.search("Matching set:", line): 
# 			currently_reading_in = ""
# 		elif re.search("Labels:", line):
# 			currently_reading_in_section = "labels"
# 		elif re.search("Image keys:", line):
# 			currently_reading_in_section = ""
# 		elif re.search("Speech keys:", line):
# 			currently_reading_in_section = "speech_keys"
# 		elif re.search("Keys:", line):
# 			currently_reading_in_section = "keys"
# 		elif len(line.strip().split()) == 0:
# 			currently_reading_in = ""
# 			currently_reading_in_section = ""
# 			continue

# 		elif currently_reading_in == "support_set" and currently_reading_in_section == "labels":
# 			line_parts = line.strip().split()
# 			episodes[curr_episode]["support_set"]["labels"].append(line_parts[0])
# 		elif currently_reading_in == "support_set" and currently_reading_in_section == "speech_keys":
# 			line_parts = line.strip().split()
# 			data, key, length = find_data_of_key(line_parts, in_data, in_keys, in_labels, in_lengths)
# 			episodes[curr_episode]["support_set"]["keys"].extend(key)
# 			episodes[curr_episode]["support_set"]["data"].extend(data)
# 			episodes[curr_episode]["support_set"]["lengths"].extend(length)

# 		elif currently_reading_in == "query" and currently_reading_in_section == "labels":
# 			line_parts = line.strip().split()
# 			episodes[curr_episode]["query"]["labels"].append(line_parts[0])
# 		elif currently_reading_in == "query" and currently_reading_in_section == "keys":
# 			line_parts = line.strip().split()
# 			data, key, length = find_data_of_key(line_parts, in_data, in_keys, in_labels, in_lengths)
# 			episodes[curr_episode]["query"]["keys"].extend(key)
# 			episodes[curr_episode]["query"]["data"].extend(data)
# 			episodes[curr_episode]["query"]["lengths"].extend(length)


# 	return episodes
# def num_of_examples_per_class_left(class_labels, labels, num_to_still_have, num_zeros_to_still_have, print_info=False):

# 	classes = set(class_labels)
# 	return_val = True

# 	for class_lab in classes:
# 		indexes = np.where(np.asarray(labels) == class_lab)[0]
# 		if print_info:
# 			print("{} remaining datapoints of label {}".format(len(indexes), class_lab))

# 		if (len(indexes) < num_to_still_have or (class_lab == "0" and len(indexes) < num_zeros_to_still_have)): return_val = False
# 	return return_val
# def num_of_examples_per_class_initial(class_labels, numbers, labels): 
# 	classes = set(class_labels)
# 	new_numbers = np.empty((len(classes)))
# 	problem_class = []


# 	for class_lab in classes:
# 		indexes = np.where(np.asarray(labels) == class_lab)[0]
# 		new_numbers[int(class_lab)] = len(indexes)
# 		if numbers.all() != 0:
# 			if class_lab == "0" and numbers[int(class_lab)] - 3 != new_numbers[int(class_lab)]: 
# 				problem_class.append(class_lab)
# 			elif class_lab != "0" and numbers[int(class_lab)] - 2 != new_numbers[int(class_lab)]: 
# 				problem_class.append(class_lab)

# 	return new_numbers, problem_class

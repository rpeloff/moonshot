

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from sklearn import manifold, preprocessing
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import subprocess
import sys
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.fftpack import dct

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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

def construct_support_set_with_keys(sp_x=None, sp_labels=None, sp_keys=None, sp_lengths=None, im_x=None, im_labels=None, im_keys=None, num_to_sample=11):
    
    #______________________________________________________________________________________________
    # Speech part
    #______________________________________________________________________________________________

    include_speech = True if sp_x is not None and sp_labels is not None and sp_lengths is not None and sp_keys is not None else False 
    if include_speech:
        # S_sp_x, S_sp_labels, S_sp_lengths, S_sp_keys, sp_x, sp_labels, sp_lengths, sp_keys = sampling_with_keys(sp_x, sp_labels, sp_keys, num_to_sample, lengths=sp_lengths)
        S_sp_x, S_sp_labels, S_sp_lengths, S_sp_keys = sampling_with_keys(sp_x, sp_labels, sp_keys, num_to_sample, lengths=sp_lengths)
        #speech = np.zeros((num_to_sample, np.max(S_sp_lengths), sp_x[0].shape[-1]))

    #______________________________________________________________________________________________
    # Image part
    #______________________________________________________________________________________________

    include_images = True if im_x is not None and im_labels is not None and im_keys is not None else False
    if include_images: 
        # S_im_x, S_im_labels, S_im_keys, im_x, im_labels, im_keys = sampling_with_keys(im_x, im_labels, im_keys, num_to_sample, S_sp_labels if include_speech is True else None)
        S_im_x, S_im_labels, S_im_keys = sampling_with_keys(im_x, im_labels, im_keys, num_to_sample, S_sp_labels if include_speech is True else None)
        #images = np.empty((num_to_sample, im_x[0].shape[-1]))
        #image_keys = np.empty((num_to_sample))

    #______________________________________________________________________________________________
    # Construct support set
    #______________________________________________________________________________________________

    support_set = {}
    image_keys = []
    image_labels = []
    images = []
    speech = []
    for i, lab in enumerate(S_sp_labels):
        speech.append(S_sp_x[i])
        im_index = np.where(np.asarray(S_im_labels) == lab)[0][0]
        images.append(S_im_x[im_index])
        image_keys.append(S_im_keys[im_index])
        image_labels.append(S_im_labels[im_index])
    support_set["images"] = images
    support_set["image_keys"] = image_keys
    support_set["image_labels"] = image_labels
    support_set["speech"] = speech
    support_set["speech_lengths"] = S_sp_lengths
    support_set["speech_keys"] = S_sp_keys
    support_set["speech_labels"] = S_sp_labels


    return support_set #, sp_x, sp_labels, sp_lengths, sp_keys, im_x, im_labels, im_keys

def construct_few_shot_support_set_with_keys(sp_x=None, sp_labels=None, sp_keys=None, sp_lengths=None, im_x=None, im_labels=None, im_keys=None, num_to_sample=11, num_of_each_sample=5,):
    
    #______________________________________________________________________________________________
    # Speech part
    #______________________________________________________________________________________________

    include_speech = True if sp_x is not None and sp_labels is not None and sp_lengths is not None and sp_keys is not None else False 
    if include_speech:
        speech_dict = sample_multiple_keys(
            sp_x, sp_labels, sp_keys, lengths=sp_lengths, num_to_sample=num_to_sample, num_of_each_sample=num_of_each_sample, do_switch=True
            )
        speech_labels = [lab for lab in speech_dict]  
    #______________________________________________________________________________________________
    # Image part
    #______________________________________________________________________________________________

    include_images = True if im_x is not None and im_labels is not None and im_keys is not None else False
    if include_images: 
        
        image_dict = sample_multiple_keys(
            im_x, im_labels, im_keys, lengths=None, num_to_sample=num_to_sample, num_of_each_sample=num_of_each_sample,  
            labels_wanted=speech_labels.copy() if include_speech is True else None, #keys_to_not_include=[], 
            do_switch=True
            )
        
    #______________________________________________________________________________________________
    # Construct support set
    #______________________________________________________________________________________________

    support_set = {}
    
    for key in speech_dict:
        support_set[key] = {}
        support_set[key]["image_data"] = image_dict[key]["data"]
        support_set[key]["image_keys"] = image_dict[key]["keys"]
        support_set[key]["speech_data"] = speech_dict[key]["data"]
        support_set[key]["speech_keys"] = speech_dict[key]["keys"]
        support_set[key]["speech_lengths"] = speech_dict[key]["lengths"]

    return support_set #, sp_x, sp_labels, sp_lengths, sp_keys, im_x, im_labels, im_keys

def sample_multiple_keys(x, labels, keys, lengths=None, num_to_sample=10, num_of_each_sample=1, labels_wanted=None, exclude_key_list=[], do_switch=False):
    
    data_dict = {}
    key_list = exclude_key_list.copy()
    # keys_to_not_include.extend(keys_list_to_not_include)

    if lengths is not None:
        for i in range(num_of_each_sample):
            x_data, x_labels, x_lengths, x_keys = sampling_with_keys(
                x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=lengths, do_switch=do_switch,
                exclude_keys=key_list
                )
            for i, label in enumerate(x_labels):
                if label not in data_dict: 
                    data_dict[label] = {}
                    data_dict[label]["data"] = []
                    data_dict[label]["lengths"] = []
                    data_dict[label]["keys"] = []
                if x_keys[i] in data_dict[label]["keys"]: 
                    print("Key already in list")
                    continue
                data_dict[label]["data"].append(x_data[i])
                data_dict[label]["lengths"].append(x_lengths[i])
                data_dict[label]["keys"].append(x_keys[i])
            key_list.extend(x_keys)
            labels_wanted = x_labels
            
    else:
        for i in range(num_of_each_sample):
            x_data, x_labels, x_keys = sampling_with_keys(
                x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=None, do_switch=do_switch,
                exclude_keys=key_list
                )
            for i, label in enumerate(x_labels):
                if label not in data_dict: 
                    data_dict[label] = {}
                    data_dict[label]["data"] = []
                    data_dict[label]["keys"] = []
                if x_keys[i] in data_dict[label]["keys"]: 
                    print("Key already in list")
                    continue
                data_dict[label]["data"].append(x_data[i])
                data_dict[label]["keys"].append(x_keys[i])
              
            key_list.extend(x_keys)
            labels_wanted = x_labels

    return data_dict

def sampling_with_keys(x, labels, keys, num_to_sample, labels_wanted=None, lengths=None, do_switch=True, exclude_keys=[]):
    np.random.seed(1)
    counter = 0
    support_set_indices = []
    support_set_x = []
    labels_to_get = (
        [] if labels_wanted is None else
        labels_wanted
        )
    support_set_labels = []
    support_set_lengths = []
    support_set_keys = []
    
    while(counter < num_to_sample):

        index = random.randint(0, len(x)-1)

        cur_label = str(labels[index])
        label_to_get = cur_label
        if cur_label == '0' and do_switch: 
            z_or_o = random.randint(0, 1)
            if z_or_o == 0:
                label_to_get = 'z' if 'z' not in support_set_labels else 'o'
            else: 
                label_to_get = 'o' if 'o' not in support_set_labels else 'z' 
            
        
        if labels_wanted is None:

            if label_to_get not in support_set_labels and index not in support_set_indices and keys[index] not in exclude_keys:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                support_set_keys.append(keys[index])
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1
        else:
            
            if label_to_get in labels_to_get and label_to_get not in support_set_labels and index not in support_set_indices and keys[index] not in exclude_keys:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                support_set_keys.append(keys[index])
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1

    if lengths is not None: 
        # x, x_labels, x_lengths, x_keys = keys_cutter(x, labels, support_set_indices, keys, lengths)
        return support_set_x, support_set_labels, support_set_lengths, support_set_keys #, x, x_labels, x_lengths, x_keys

    else: 
        # x, x_labels, x_keys = keys_cutter(x, labels, support_set_indices, keys)
        return support_set_x, support_set_labels, support_set_keys #, x, x_labels, x_keys

def keys_cutter(x, labels, indices, keys, lengths=None):
    indices = sorted(indices)
    cutted_x = []
    cutted_labels = []
    cutted_lengths = []
    cutted_keys = []
    last_index = 0

    for index in sorted(indices):
        if index >= len(x):
            print("Invalid index")
        elif index >= last_index:

            cutted_x.extend(x[last_index: index])
            cutted_labels.extend(labels[last_index: index])
            cutted_keys.extend(keys[last_index: index])
            if lengths is not None: cutted_lengths.extend(lengths[last_index: index])
            last_index = index + 1
            
    cutted_x.extend(x[last_index: ])
    cutted_labels.extend(labels[last_index: ])
    cutted_keys.extend(keys[last_index: ])
    if lengths is not None: cutted_lengths.extend(lengths[last_index: ])


    if lengths is not None: return cutted_x, cutted_labels, cutted_lengths, cutted_keys
    else: return cutted_x, cutted_labels, cutted_keys 

# def sample_multiple_keys(x, labels, keys, lengths=None, num_to_sample=10, num_of_each_sample=1, keys_to_not_include=[], labels_wanted=None, do_switch=False):
    
#     data_dict = {}
#     # labels_wanted = None

#     if lengths is not None:
#         for i in range(num_of_each_sample):
#             x_data, x_labels, x_lengths, x_keys = sampling_with_keys(
#                 x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=lengths, do_switch=do_switch,
#                 keys_to_not_include=keys_to_not_include
#                 )
#             for i, label in enumerate(x_labels):
#                 if label not in data_dict: 
#                     data_dict[label] = {}
#                     data_dict[label]["data"] = []
#                     data_dict[label]["lengths"] = []
#                     data_dict[label]["keys"] = []
#                 if x_keys[i] in data_dict[label]["keys"]: 
#                     print("Key already in list")
#                     continue
#                 data_dict[label]["data"].append(x_data[i])
#                 data_dict[label]["lengths"].append(x_lengths[i])
#                 data_dict[label]["keys"].append(x_keys[i])
#             keys_to_not_include.extend(x_keys)
#             labels_wanted = x_labels
#     else: 
#         for i in range(num_of_each_sample):
#             print(keys_to_not_include)
#             print(labels_wanted)
#             x_data, x_labels, x_keys = sampling_with_keys(
#                 x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=None, do_switch=do_switch,
#                 keys_to_not_include=keys_to_not_include
#                 )
#             for i, label in enumerate(x_labels):
#                 if label not in data_dict: 
#                     data_dict[label] = {}
#                     data_dict[label]["data"] = []
#                     data_dict[label]["keys"] = []
#                 if x_keys[i] in data_dict[label]["keys"]: 
#                     print("Key already in list")
#                     continue
#                 data_dict[label]["data"].append(x_data[i])
#                 data_dict[label]["keys"].append(x_keys[i])
              
#             keys_to_not_include.extend(x_keys)
#             labels_wanted = x_labels
#         print(keys_to_not_include)
#         print(labels_wanted)
#         print(len(data_dict))

#     return data_dict

def sample_query_or_matching_set(x, labels, keys, lengths=None, num_to_sample=10, num_of_each_sample=1, keys_to_not_include=[]):

    data_dict = {}

    if lengths is not None: 
        # x_data, x_labels, x_lengths, x_keys, x, labels, lengths, keys = sampling_with_keys(
        #     x, labels, keys, num_to_sample, labels_wanted=None, lengths=lengths, do_switch=False,
        #     keys_to_not_include=keys_to_not_include
        #     )
        x_data, x_labels, x_lengths, x_keys = sampling_with_keys(
            x, labels, keys, num_to_sample, labels_wanted=None, lengths=lengths, do_switch=False,
            keys_to_not_include=keys_to_not_include
            )

        data_dict["data"] = x_data
        data_dict["labels"] = x_labels
        data_dict["keys"] = x_keys
        data_dict["lengths"] = x_lengths

        return data_dict#, x, labels, lengths, keys

    else: 
        # x_data, x_labels, x_keys, x, labels, keys = sampling_with_keys(
        #     x, labels, keys, num_to_sample, labels_wanted=None, lengths=None, do_switch=False,
        #     keys_to_not_include=keys_to_not_include
        #     )
        x_data, x_labels, x_keys = sampling_with_keys(
            x, labels, keys, num_to_sample, labels_wanted=None, lengths=None, do_switch=False,
            keys_to_not_include=keys_to_not_include
            )
        print(len(x_data), x_labels, x_keys)
        data_dict["data"] = x_data
        data_dict["labels"] = x_labels
        data_dict["keys"] = x_keys

        return data_dict#, x, labels, keys


def image_support_set(im_x, im_labels):

    support_set, im_x, im_labels = construct_support_set(None, None, None, im_x, im_labels, 11)

    return support_set, im_x, im_labels

def speech_support_set(sp_x, sp_labels, sp_lengths):
    
    support_set, sp_x, sp_labels, sp_lengths  = construct_support_set(sp_x, sp_labels, sp_lengths, None, None, 11)
    
    return support_set, sp_x, sp_labels, sp_lengths 

def get_support_set(speech_path, image_path):

    #______________________________________________________________________________________________
    # Speech data
    #______________________________________________________________________________________________
    
    sp_x, sp_labels, sp_lengths, sp_keys = (
        data_library.load_speech_data_from_npz(path.join(
                feats_path, "TIDigits", "Subsets", "Words", "mfcc", "samediff_val_mfcc.npz"
                )
            )
        )
    max_frames = 100
    d_frame = 13

    print("\nLimiting dimensionality: {}".format(d_frame))
    print("Limiting number of frames: {}\n".format(max_frames))
    data_library.truncate_data_dim(sp_x, sp_lengths, d_frame, max_frames)

    #______________________________________________________________________________________________
    # Image data
    #______________________________________________________________________________________________
    if image_path == "mnist":
        out_dir = util_library.saving_path(data_path, "MNIST")
        from tensorflow.examples.tutorials.mnist import input_data   
        mnist = input_data.read_data_sets(out_dir, one_hot=True) 

        im_x, im_labels = mnist.train.next_batch(mnist.validation.num_examples)
        im_labels = np.argmax(im_labels, axis=1)

    support_set, sp_x, sp_labels, sp_lengths, im_x, im_labels = construct_support_set(sp_x, sp_labels, sp_lengths, im_x, im_labels, 11)
    
    return support_set, sp_x, sp_labels, sp_lengths, im_x, im_labels

def construct_support_set(sp_x=None, sp_labels=None, sp_lengths=None, im_x=None, im_labels=None, num_to_sample=11):
    
    #______________________________________________________________________________________________
    # Speech part
    #______________________________________________________________________________________________

    include_speech = True if sp_x is not None and sp_labels is not None and sp_lengths is not None else False 
    if include_speech:
        S_sp_x, S_sp_labels, S_sp_lengths, sp_x, sp_labels, sp_lengths = sampling(sp_x, sp_labels, num_to_sample, lengths=sp_lengths)
        speech = np.zeros((num_to_sample, np.max(S_sp_lengths), sp_x[0].shape[-1]))

    #______________________________________________________________________________________________
    # Image part
    #______________________________________________________________________________________________

    include_images = True if im_x is not None and im_labels is not None else False
    if include_images: 
        S_im_x, S_im_labels, im_x, im_labels = sampling(im_x, im_labels, num_to_sample, S_sp_labels if include_speech is True else None)
        images = np.empty((num_to_sample, im_x.shape[-1]))

    #______________________________________________________________________________________________
    # Construct support set
    #______________________________________________________________________________________________

    support_set = {}

    if include_images and include_speech:
        for i, lab in enumerate(S_sp_labels):
            speech[i, :S_sp_lengths[i], :] = S_sp_x[i]
            images[i, :] = S_im_x[ np.where(np.asarray(S_im_labels) == lab)[0][0] ]
        support_set["images"] = images
        support_set["speech"] = speech
        support_set["speech_lengths"] = S_sp_lengths
        support_set["labels"] = S_sp_labels

        return support_set, sp_x, sp_labels, sp_lengths, im_x, im_labels

    elif include_images:
        for i, lab in enumerate(S_im_labels):
            images[i, :] = S_im_x[i]
        support_set["images"] = images
        support_set["labels"] = S_im_labels

        return support_set, im_x, im_labels

    elif include_speech:
        for i, lab in enumerate(S_sp_labels):
            speech[i, :S_sp_lengths[i], :] = S_sp_x[i]
        support_set["speech"] = speech
        support_set["speech_lengths"] = S_sp_lengths
        support_set["labels"] = S_sp_labels

        return support_set, sp_x, sp_labels, sp_lengths    




def sampling(x, labels, num_to_sample, labels_wanted=None, lengths=None):

    counter = 0
    support_set_indices = []
    support_set_x = []
    labels_to_get = (
        [] if labels_wanted is None else
        labels_wanted
        )
    support_set_labels = []
    support_set_lengths = []
    
    while(counter < num_to_sample):

        index = random.randint(0, len(x)-1)

        cur_label = str(labels[index])
        label_to_get = cur_label
        if cur_label == '0': 
            z_or_o = random.randint(0, 1)
            if z_or_o == 0:
                label_to_get = 'z' if 'z' not in support_set_labels else 'o'
            else: 
                label_to_get = 'o' if 'o' not in support_set_labels else 'z' 
        
        if labels_wanted is None:

            if label_to_get not in support_set_labels:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1
        else:

            if label_to_get in labels_to_get and label_to_get not in support_set_labels:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1

    support_set_indices = sorted(support_set_indices, reverse=True)
   
    for i, ind in enumerate(support_set_indices):
        if lengths is not None:
            x, labels, lengths = speech_cutter(x, labels, ind, lengths)
        else: x, labels = image_cutter(x, labels, ind)

    if lengths is not None: return support_set_x, support_set_labels, support_set_lengths, x, labels, lengths
    else: return support_set_x, support_set_labels, x, labels

def speech_cutter(x, labels, index, lengths):
    if index >= len(x):
        print("Invalid index")
        return None
    else:
        x_1 = x[0: index]
        x_1.extend(x[index+1:])
        labels_1 = labels[0: index]
        labels_1.extend(labels[index+1:])
        lengths_1 = lengths[0: index]
        lengths_1.extend(lengths[index+1:])
        return x_1, labels_1, lengths_1 

def image_cutter(x, labels, index):
    if index >= len(x):
        print("Invalid index")
        return None
    else:
        x_1 = x[0: index, :]
        x_1 = np.concatenate((x_1, x[index+1:, :]), axis=0)
        labels_1 = labels[0: index]
        labels_1 = np.concatenate((labels_1, labels[index+1:]), axis=0)

        return x_1, labels_1

def one_shot_val(latent, val_x, val_labels, shuffle_batches_every_epoch, model_fn, support_labels):
    np.random.seed(lib["rnd_seed"])

    val_batch_iterator = batching_library.image_iterator(
        val_x, len(val_x), shuffle_batches_every_epoch=False
        )
    labels = [val_labels[i] for i in val_batch_iterator.indices]
    saver = tf.train.Saver()
    with tf.Session() as sesh:
        saver.restore(sesh, model_fn)

        for feats in val_batch_iterator:
            lat = sesh.run(
                [latent], feed_dict={X: feats}
                )[0]

        support_lat = sesh.run(
            [latent], feed_dict={X: support_set}
            )[0]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
        support_lat, support_labels)
    distances, indices = nbrs.kneighbors(lat)
   

def format(model_arch, embeddings):

    if model_arch == "rnn":

        lab = []
        for i, key in enumerate(embeddings):
            this_embed = np.reshape(embeddings[key], (1, embeddings[key].shape[0]))
            

            if i == 0: 
                embed = np.array(this_embed) 
            else:  
                embed = np.concatenate((embed, this_embed)) 
            lab.append((key.split("_")[-1])) 

    else: 
        embed = embeddings["embeddings"]
        lab = np.array(embeddings["labels"])

    return embed, lab

def eval(S, S_labels, x, x_labels):

    N = len(x)

    distances = cdist(x, S, "cosine")
    print(distances.shape)
    print(distances)

    indexes = np.argmin(distances)

def get_features(model_fn, x, x_labels, x_lengths=None):
    
    model_name = model_fn.split("/")[-2]
    lib = model_setup_library.restore_lib(path.join(model_fn, model_name + "_lib.pkl"))

    lib["train_model"] = lib["test_model"] = lib["do_clustering_etc"] = False

    if lib["training_on"] == "speech": 
        input_embed, output_embed, latent_embed = speech_model_library.rnn_speech_model(
            lib, x, x_labels, x_lengths
            )
    elif lib["training_on"] == "images": 
        if lib["architecture"] == "cnn": 
            input_embed, output_embed, latent_embed = vision_model_library.cnn_vision_model(lib, x, x_labels)
        elif lib["architecture"] == "fc": 
            input_embed, output_embed, latent_embed = vision_model_library.fc_vision_model(lib, x, x_labels)

    return input_embed, output_embed, latent_embed


        
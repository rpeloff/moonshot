"""Functions for plotting pretty pictures.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: June 2019

Some code sourced from Aurélien Geron's awesome Hands-On Machine Learning 2 examples:
https://github.com/ageron/handson-ml2
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from collections.abc import Iterable


import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig_id, path="figures", tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.exists(path):
        os.makedirs(path)
    fig_path = os.path.join(path, "{}.{}".format(fig_id, fig_extension))
    print("Saving figure", fig_path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


def plot_support_set(support_set_data, support_set_labels, k_shot, l_way, speakers=None, fig_id="support_set", cmap="binary", origin="lower"):
    # get max x and y length for axis sharing (on length basis)
    max_y, max_x = 0., 0.
    for query in support_set_data:
        query_shape = np.shape(query)
        max_y = max(query_shape[0], max_y)
        max_x = max(query_shape[1], max_x)
    # calc number of rows and cols and plot images
    n_rows = k_shot
    n_cols = l_way
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for col in range(n_cols):
        for row in range(n_rows):
            plt_index = n_cols * row + col
            img_index = n_rows * col + row
            try:
                plt.subplot(n_rows, n_cols, plt_index + 1)
                plt.imshow(support_set_data[img_index], cmap=cmap, interpolation="nearest", origin="upper")
                title = support_set_labels[img_index]
                if speakers is not None:
                    title += " ({})".format(speakers[img_index])
                plt.title(title, fontsize=12)
                plt.axis("off")
                # center plot with same x- and y-axis length across examples
                axs = plt.gca()
                y_len, x_len = np.shape(support_set_data[img_index])
                axs.set_xlim(round(-(max_x - x_len)/2), x_len + round((max_x - x_len)/2))
                if origin == "lower":
                    axs.set_ylim(round(-(max_y- y_len)/2), y_len + round((max_y - y_len)/2))
                else:
                    axs.set_ylim(y_len + round((max_y - y_len)/2), round(-(max_y- y_len)/2))
            except IndexError:
                plt.axis("off")
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig(fig_id, tight_layout=False)
    plt.show()


def plot_query_set(query_set_data, query_set_labels, n_queries, speakers=None, max_cols=15, fig_id="query_set", cmap="binary", origin="lower"):
    # get max x and y length for axis sharing (on length basis)
    max_y, max_x = 0., 0.
    for query in query_set_data:
        query_shape = np.shape(query)
        max_y = max(query_shape[0], max_y)
        max_x = max(query_shape[1], max_x)
    # calc number of rows and cols and plot images
    n_rows = int(np.ceil(n_queries / max_cols))
    n_cols = min(n_queries, max_cols)
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            img_index = n_cols * row + col
            try:
                if img_index < n_queries:
                    plt.subplot(n_rows, n_cols, img_index + 1)
                    plt.imshow(query_set_data[img_index], cmap=cmap, interpolation="nearest")
                    plt.title(query_set_labels[img_index], fontsize=12)
                    title = query_set_labels[img_index]
                    if speakers is not None:
                        title += " ({})".format(speakers[img_index])
                    plt.title(title, fontsize=12)
                    plt.axis("off")
                    # center plot with same x- and y-axis length across examples
                    axs = plt.gca()
                    y_len, x_len = np.shape(query_set_data[img_index])
                    axs.set_xlim(round(-(max_x - x_len)/2), x_len + round((max_x - x_len)/2))
                    if origin == "lower":
                        axs.set_ylim(round(-(max_y- y_len)/2), y_len + round((max_y - y_len)/2))
                    else:
                        axs.set_ylim(y_len + round((max_y - y_len)/2), round(-(max_y- y_len)/2))
            except IndexError:
                plt.axis("off")
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig(fig_id, tight_layout=False)
    plt.show()


def plot_multimodal_support_set(image_support_set_data, image_support_set_labels,
                                speech_support_set_data, speech_support_set_labels,
                                k_shot, l_way, speakers=None, fig_id="support_set",
                                image_cmap="binary", speech_cmap="inferno",
                                image_origin="lower", speech_origin="lower"):
    # get max x and y speech length for axis sharing (on length basis)
    speech_max_y, speech_max_x = 0., 0.
    for query in speech_support_set_data:
        query_shape = np.shape(query)
        speech_max_y = max(query_shape[0], speech_max_y)
        speech_max_x = max(query_shape[1], speech_max_x)
    # calc number of rows and cols and plot images
    n_rows = k_shot
    n_cols = l_way
    plt.figure(figsize=(n_cols * 1.2 * 2, n_rows * 1.2))
    for col in range(n_cols):
        for row in range(n_rows):
            plt_index = n_cols * row + col
            img_index = n_rows * col + row
            try:
                plt.subplot(n_rows, n_cols * 2, plt_index * 2 + 1)
                plt.imshow(image_support_set_data[img_index], interpolation="nearest", cmap=image_cmap, origin=image_origin)
                plt.title(image_support_set_labels[img_index], fontsize=12)
                plt.axis("off")
                plt.subplot(n_rows, n_cols * 2, plt_index * 2 + 1 + 1)
                plt.imshow(speech_support_set_data[img_index], interpolation="nearest", cmap=speech_cmap, origin=speech_origin)
                title = speech_support_set_labels[img_index]
                if speakers is not None:
                    title += " ({})".format(speakers[img_index])
                plt.title(title, fontsize=12)
                plt.axis("off")
                # center plot speech with same x- and y-axis length across examples
                axs = plt.gca()
                y_len, x_len = np.shape(speech_support_set_data[img_index])
                axs.set_xlim(round(-(speech_max_x - x_len)/2), x_len + round((speech_max_x - x_len)/2))
                if speech_origin == "lower": 
                    axs.set_ylim(round(-(speech_max_y- y_len)/2), y_len + round((speech_max_y - y_len)/2))
                else:
                    axs.set_ylim(y_len + round((speech_max_y - y_len)/2), round(-(speech_max_y- y_len)/2))
            except IndexError:
                plt.axis("off")
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig(fig_id, tight_layout=False)
    plt.show()


def transpose_variable_array(variable_arr):
    tranpose_array = lambda arr: np.transpose(arr, axes=[1, 0])
    return list(map(tranpose_array, variable_arr))

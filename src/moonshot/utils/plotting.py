"""Functions for plotting pretty pictures.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: June 2019

Some code sourced from Aurélien Geron's incredible Hands-On Machine Learning 2
examples: https://github.com/ageron/handson-ml2
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import logging


import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np


from moonshot.utils import file_io


def save_figure(filename, path="figures", figure=None,
                tight_layout=False, fig_extension="png", resolution=300):
    """Save current plot or specified figure to disk."""
    file_io.check_create_dir(path)

    logging.log(
        logging.INFO,
        "Saving figure '{}' to directory: {}".format(filename, path))

    if figure is None:  # fetch default plot (only works before plt.show)
        figure = plt

    if tight_layout:
        figure.tight_layout()

    fig_path = os.path.join(path, filename)
    figure.savefig(fig_path, format=fig_extension, dpi=resolution)


# TODO(rpeloff) old code, remove if not using

# def transpose_variable_array(variable_arr):
#     tranpose_array = lambda arr: np.transpose(arr, axes=[1, 0])
#     return list(map(tranpose_array, variable_arr))


# def plot_support_set(support_set_data, support_set_labels, k_shot, l_way, speakers=None, fig_id="support_set", cmap="binary", origin="lower"):
#     # get max x and y length for axis sharing (on length basis)
#     max_y, max_x = 0., 0.
#     for query in support_set_data:
#         query_shape = np.shape(query)
#         max_y = max(query_shape[0], max_y)
#         max_x = max(query_shape[1], max_x)
#     # calc number of rows and cols and plot images
#     n_rows = k_shot
#     n_cols = l_way
#     plt.figure(figsize=(n_cols * 1.3, n_rows * 1.3))
#     for col in range(n_cols):
#         for row in range(n_rows):
#             plt_index = n_cols * row + col
#             img_index = n_rows * col + row
#             try:
#                 plt.subplot(n_rows, n_cols, plt_index + 1)
#                 plt.imshow(support_set_data[img_index], cmap=cmap, interpolation="nearest", origin="upper")
#                 title = support_set_labels[img_index]
#                 if speakers is not None:
#                     title += " ({})".format(speakers[img_index])
#                 plt.title(title, fontsize=12)
#                 plt.axis("off")
#                 # center plot with same x- and y-axis length across examples
#                 axs = plt.gca()
#                 shape = np.shape(support_set_data[img_index])
#                 y_len, x_len = shape[0], shape[1]
#                 axs.set_xlim(round(-(max_x - x_len)/2), x_len + round((max_x - x_len)/2))
#                 if origin == "lower":
#                     axs.set_ylim(round(-(max_y- y_len)/2), y_len + round((max_y - y_len)/2))
#                 else:
#                     axs.set_ylim(y_len + round((max_y - y_len)/2), round(-(max_y- y_len)/2))
#             except IndexError:
#                 plt.axis("off")
#     plt.subplots_adjust(wspace=0.2, hspace=0.5)  # spacing between subplots
#     if fig_id is not None:
#         save_fig(fig_id, tight_layout=False)
#     plt.show()


# def plot_query_set(query_set_data, query_set_labels, n_queries, speakers=None, max_cols=15, fig_id="query_set", cmap="binary", origin="lower"):
#     # get max x and y length for axis sharing (on length basis)
#     max_y, max_x = 0., 0.
#     for query in query_set_data:
#         query_shape = np.shape(query)
#         max_y = max(query_shape[0], max_y)
#         max_x = max(query_shape[1], max_x)
#     # calc number of rows and cols and plot images
#     n_rows = int(np.ceil(n_queries / max_cols))
#     n_cols = min(n_queries, max_cols)
#     plt.figure(figsize=(n_cols * 1.3, n_rows * 1.3))
#     for row in range(n_rows):
#         for col in range(n_cols):
#             img_index = n_cols * row + col
#             try:
#                 if img_index < n_queries:
#                     plt.subplot(n_rows, n_cols, img_index + 1)
#                     plt.imshow(query_set_data[img_index], cmap=cmap, interpolation="nearest")
#                     plt.title(query_set_labels[img_index], fontsize=12)
#                     title = query_set_labels[img_index]
#                     if speakers is not None:
#                         title += " ({})".format(speakers[img_index])
#                     plt.title(title, fontsize=12)
#                     plt.axis("off")
#                     # center plot with same x- and y-axis length across examples
#                     axs = plt.gca()
#                     shape = np.shape(query_set_data[img_index])
#                     y_len, x_len = shape[0], shape[1]
#                     axs.set_xlim(round(-(max_x - x_len)/2), x_len + round((max_x - x_len)/2))
#                     if origin == "lower":
#                         axs.set_ylim(round(-(max_y- y_len)/2), y_len + round((max_y - y_len)/2))
#                     else:
#                         axs.set_ylim(y_len + round((max_y - y_len)/2), round(-(max_y- y_len)/2))
#             except IndexError:
#                 plt.axis("off")
#     plt.subplots_adjust(wspace=0.2, hspace=0.5)  # spacing between subplots
#     if fig_id is not None:
#         save_fig(fig_id, tight_layout=False)
#     plt.show()


# def plot_multimodal_support_set(image_support_set_data, image_support_set_labels,
#                                 speech_support_set_data, speech_support_set_labels,
#                                 k_shot, l_way, speakers=None, fig_id="support_set",
#                                 image_cmap="binary", speech_cmap="inferno",
#                                 image_origin="lower", speech_origin="lower"):
#     # get max x and y speech length for axis sharing (on length basis)
#     speech_max_y, speech_max_x = 0., 0.
#     for query in speech_support_set_data:
#         query_shape = np.shape(query)
#         speech_max_y = max(query_shape[0], speech_max_y)
#         speech_max_x = max(query_shape[1], speech_max_x)
#     # calc number of rows and cols and plot images
#     n_rows = k_shot
#     n_cols = l_way
#     plt.figure(figsize=(n_cols * 1.3 * 2, n_rows * 1.3))
#     for col in range(n_cols):
#         for row in range(n_rows):
#             plt_index = n_cols * row + col
#             img_index = n_rows * col + row
#             try:
#                 plt.subplot(n_rows, n_cols * 2, plt_index * 2 + 1)
#                 plt.imshow(image_support_set_data[img_index], interpolation="nearest", cmap=image_cmap, origin=image_origin)
#                 plt.title(image_support_set_labels[img_index], fontsize=12)
#                 plt.axis("off")
#                 plt.subplot(n_rows, n_cols * 2, plt_index * 2 + 1 + 1)
#                 plt.imshow(speech_support_set_data[img_index], interpolation="nearest", cmap=speech_cmap, origin=speech_origin)
#                 title = speech_support_set_labels[img_index]
#                 if speakers is not None:
#                     title += " ({})".format(speakers[img_index])
#                 plt.title(title, fontsize=12)
#                 plt.axis("off")
#                 # center plot speech with same x- and y-axis length across examples
#                 axs = plt.gca()
#                 shape = np.shape(speech_support_set_data[img_index])
#                 y_len, x_len = shape[0], shape[1]
#                 axs.set_xlim(round(-(speech_max_x - x_len)/2), x_len + round((speech_max_x - x_len)/2))
#                 if speech_origin == "lower": 
#                     axs.set_ylim(round(-(speech_max_y- y_len)/2), y_len + round((speech_max_y - y_len)/2))
#                 else:
#                     axs.set_ylim(y_len + round((speech_max_y - y_len)/2), round(-(speech_max_y- y_len)/2))
#             except IndexError:
#                 plt.axis("off")
#     plt.subplots_adjust(wspace=0.2, hspace=0.5)  # spacing between subplots
#     if fig_id is not None:
#         save_fig(fig_id, tight_layout=False)
#     plt.show()


# def plot_predictions(support_set_data, support_set_labels, query_predict_idx, query_distances,
#                      k_shot, l_way, speakers=None, fig_id="predictions", cmap="binary", origin="lower"):
#     # get max x and y length for axis sharing (on length basis)
#     max_y, max_x = 0., 0.
#     for query in support_set_data:
#         query_shape = np.shape(query)
#         max_y = max(query_shape[0], max_y)
#         max_x = max(query_shape[1], max_x)
#     # get top-5 closest distance indices
#     top_5_idx = np.argsort(query_distances)[:5]
#     # calc number of rows and cols and plot images
#     n_rows = k_shot
#     n_cols = l_way
#     plt.figure(figsize=(n_cols * 1.8, n_rows * 1.7))
#     for col in range(n_cols):
#         for row in range(n_rows):
#             plt_index = n_cols * row + col
#             img_index = n_rows * col + row
#             try:
#                 plt.subplot(n_rows, n_cols, plt_index + 1)
#                 axs = plt.gca()
#                 plt.imshow(support_set_data[img_index], cmap=cmap, interpolation="nearest", origin="upper")
#                 title = str(support_set_labels[img_index])
#                 if speakers is not None:
#                     title += " ({})".format(speakers[img_index])
#                 plt.title(title, fontsize=12)
#                 axs.set_xlabel("cost: {:.4f}".format(query_distances[img_index]))
#                 # hide axes but keep x-label
#                 axs.yaxis.set_visible(False)  # set y-axis invisible
#                 plt.setp(axs.spines.values(), visible=False)  # make spines (the box) invisible
#                 axs.tick_params(bottom=False, labelbottom=False)  # remove ticks and labels for y-axis
#                 axs.patch.set_visible(False)  # remove background patch if non-white background
#                 # center plot with same x- and y-axis length across examples
#                 shape = np.shape(support_set_data[img_index])
#                 y_len, x_len = shape[0], shape[1]
#                 axs.set_xlim(round(-(max_x - x_len)/2), x_len + round((max_x - x_len)/2))
#                 if origin == "lower":
#                     axs.set_ylim(round(-(max_y- y_len)/2), y_len + round((max_y - y_len)/2))
#                 else:
#                     axs.set_ylim(y_len + round((max_y - y_len)/2), round(-(max_y- y_len)/2))
#                 # create a patch to show predictions
#                 if img_index in top_5_idx:
#                     rect = patches.Rectangle(
#                         (0, 0), x_len, y_len,
#                         linewidth=5, edgecolor="g", facecolor="none")
#                     axs.add_patch(rect)
#                     plt.title(title, fontsize=12, fontweight="bold", color="g")
#                 if img_index in query_predict_idx:
#                     rect = patches.Rectangle(
#                         (0, 0), x_len, y_len,
#                         linewidth=5, edgecolor="r", facecolor="none")
#                     axs.add_patch(rect)
#                     plt.title(title, fontsize=12, fontweight="bold", color="r")

#             except IndexError:
#                 plt.axis("off")
#     plt.subplots_adjust(wspace=0.2, hspace=0.5)  # spacing between subplots
#     if fig_id is not None:
#         save_fig(fig_id, tight_layout=False)
#     plt.show()


# def plot_flickr_audio_keyword_images(
#         flickr_image_set, image_uids, keyword,
#         max_per_row=6, image_size=(3, 3), interpolation="lanczos"):
#     """Plot Flickr 8k images for given keyword and corresponding image UIDs."""
#     n_images = len(image_uids)
#     n_cols = min(n_images, max_per_row)
#     n_rows = int(np.ceil(n_images / max_per_row))
#     plt.figure(figsize=(n_cols * image_size[0], n_rows * image_size[1]))
#     plt.suptitle(keyword, fontsize=14)
#     for image_index, uid in enumerate(image_uids):
#         plt.subplot(n_rows, n_cols, image_index + 1)
#         plt.imshow(
#             flickr_image_set[1][np.where(flickr_image_set[0] == uid)[0][0]],
#             interpolation="lanczos")
#         plt.title(uid)
#         plt.axis("off")
#     plt.subplots_adjust(wspace=0.2, hspace=0.5)  # spacing between subplots
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust subplots area for suptitle
#     figure = plt.gcf()  # get figure before showing which creates new figure
#     plt.show()
#     return figure

"""MOONSHOT setup script.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import setuptools


# requirements for running moonshot
moonshot_requires = [      
    "absl-py",
    "numpy",
    "scipy",
    "six",
    "matplotlib",
    "scikit-learn",
    "scikit-image",
]

# requirements for running baseline models in moonshot/baselines
baselines_requires = [
    "numba",
    "tensorflow-gpu",
]

# requirements for extracting speech feature and
# filtering one-shot keywords (specific versions for reproducibility)
extraction_requires = [
    "nltk==3.4.5",
    "spacy==2.1.8",
]

# shell command line tools
scripts = [
    # os.path.join("src", "moonshot", "features", "ms-apply-cmvn-dd"),
    # os.path.join("src", "moonshot", "features", "ms-extract-speech-features"),
]

# package command line tools
package_scripts = [
    # "ms-kaldi-to-numpy=moonshot.features.kaldi_to_numpy:main",
    # "ms-prepare-tidigits-segments=moonshot.features.tidigits.tidigits_segments_prep:main",
    # "ms-prepare-flickr-vad=moonshot.features.flickr_audio.flickr8k_data_prep_vad:main",
    # "ms-get-flickr-fbank=moonshot.features.flickr_audio.get_kaldi_fbank:main",
    # "ms-get-flickr-mfcc=moonshot.features.flickr_audio.get_kaldi_mfcc:main",
    # "ms-get-flickr-words=moonshot.features.flickr_audio.get_iso_words:main",
    # "ms-filter-flickr-keywords=moonshot.features.flickr_audio.write_filtered_sets:main",
]

# get long description from readme
with open("README.md", "r") as file:
    long_description = file.read()

# make this project pip installable with `pip install -e .`
setuptools.setup(
    name="moonshot",
    version="0.0.1",
    author="Ryan Eloff",
    author_email="ryan.peter.eloff@gmail.com",
    description=("MOONSHOT: MultimOdal ONe-SHOT learning benchmark",
                 "Reproducible and comparable experiments for multimodal one-shot learning"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpeloff/multimodal_one_shot_benchmark",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    scripts=scripts,
    entry_points={"console_scripts": package_scripts},
    install_requires=moonshot_requires,
    extras_require={
        "baselines": baselines_requires,
        "extraction": extraction_requires,
    },
    test_suite='nose.collector',
    # see https://pypi.org/classifiers for list of classifiers
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

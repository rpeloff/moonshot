
# Setup and build scripts.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: September 2019

ifeq ($(shell id -u -r), 0)
SUDO_CMD :=
else
SUDO_CMD := sudo
endif

.PHONY: moonshot moonshot_develop baselines extraction clean features test test_debug upgrade_tools

moonshot:
	make upgrade_tools \
	&& python3 -m pip install .

moonshot_develop:
	make upgrade_tools \
	&& python3 -m pip install -e .

baselines:
	make upgrade_tools \
	&& python3 -m pip install -e .[baselines]

extraction:
	make upgrade_tools \
	&& python3 -m pip install -e .[extraction] \
	&& python3 -c "import nltk; nltk.download('stopwords')" \
	&& python3 -m spacy download -d en_core_web_lg-2.1.0

clean:
	rm -rf data/processed/tidigits/* data/processed/flickr_audio/* data/logs/tidigits/* data/logs/flickr_audio/* 

features:
	ms-extract-speech-features

test:
	make upgrade_tools \
	&& python3 -m pip install pytest \
	&& python3 -m pytest tests/

test_debug:
	make upgrade_tools \
	&& python3 -m pip install pytest \
	&& python3 -m pytest --pdb --maxfail=3 tests/ 

upgrade_tools:
	# run with sudo if not running as root
	$(SUDO_CMD) apt-get update -y \
    && $(SUDO_CMD) apt-get install -y --no-install-recommends python3-pip \
    && python3 -m pip install --upgrade setuptools wheel

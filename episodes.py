"""Functions to generate few-shot episodes.

Author: Ryan Eloff && Leanne Nortje
Contact: ryan.peter.eloff@gmail.com
Date: June 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def sample_episode(uids, labels, speakers=None,
				   n_queries=15, k_shot=1, l_way=5, shuffle=True,
				   speaker_mode="baseline"):
	"""Sample a single K-shot L-way episode support set and N test queries.

	Speaker mode one of ["baseline", "easy", "hard", "distractor"]
	- baseline (normal) tests:
	    No speaker rules, speaker repetition allowed in and between sets.
	- easy tests (might be useless):
	    Support speakers same as corresponding query set speakers for each given class.
	- hard tests:
	    Support set speakers different to corresponding query set speakers; query and support set speakers are disjoint.
	- distractor tests (in case DTW hard test == baseline tests):
	    Similar to hard tests, except query set speakers appear in support set for other classes known as distractors.

	Speaker mode examples
	- "baseline":
	    e.g. classes [1,2,3,4,5], speakers [a,b,...,z]
	    sample query speakers {1->a, 2->z, 3->f, 4->f, 5->m}
	    sample support speakers {1->z, 2->h, 3->f, 4->b, 5->e}
	- "easy":
	    e.g. classes [1,2,3,4,5], speakers [a,b,...,z]
	    sample query speakers {1->a, 2->b, 3->c, 4->d, 5->e}
	    sample support speakers {1->a, 2->b, 3->c, 4->d, 5->e}
	- "hard":
	    e.g. classes [1,2,3,4,5], speakers [a,b,...,z]
	    sample query speakers {1->a, 2->b, 3->c, 4->d, 5->e}
	    sample support speakers {1->f, 2->m, 3->z, 4->k, 5->g}
	- "distractor":
	    e.g. classes [1,2,3,4,5], speakers [a,b,...,z]
	    sample query speakers {1->a, 2->b, 3->c, 4->d, 5->e}
	    sample support speakers {1->b, 2->c, 3->d, 4->e, 5->a]}

	Speaker mode hypothesis
	- DTW model:
	    easy > baseline > hard > distractor
	- Neural models:
	    baseline == easy == hard == distractor
	"""
	# copy labels and uids and create shuffle indices (so that first instance of duplicate uids is different to previous episodes)
	uids = np.asarray(uids).copy()
	labels = np.asarray(labels).copy()
	permute_idx = np.random.permutation(len(uids))
	# get unique class labels
	classes = np.unique(labels)
	# choose L unique class labels for this episode
	episode_cls = np.random.choice(classes, size=l_way, replace=False)
	# choose N query labels from the chosen episode class labels with replacement (uniform distribution among the N labels)
	query_cls = episode_cls[np.random.choice(len(episode_cls), size=n_queries, replace=True)]
	# copy speakers and get unique speaker ids if speakers specified
	if speakers is not None:
		speakers = np.asarray(speakers).copy()
		unique_speakers = np.unique(speakers)
		# sample L unique speaker uids for the episode classes if using "distractor" speaker mode
		if speaker_mode == "distractor":
			episode_speakers = unique_speakers[np.random.choice(len(unique_speakers), size=l_way, replace=False)]
    # sample support and query set
	support_idx, query_idx = [], []
	speaker_exclusion, query_speaker_exclusion = [], []
	for cls_idx, cls in enumerate(episode_cls):
		# set initial valid indices from which to sample as the shuffled indices
		valid_idx = permute_idx
		# get indices where dataset labels match the current class and cut valid indices
		label_cls_idx = np.where(labels == cls)[0]
		valid_idx = np.intersect1d(valid_idx, label_cls_idx)
		# get number of episode query labels which match the current class
		num_cls_queries = len(np.where(query_cls == cls)[0])
		# get indices of unique uids (removing possible repetitions) and cut valid indices
		uids_unique_cls_idx = np.unique(uids, return_index=True)[1]
		valid_idx = np.intersect1d(valid_idx, uids_unique_cls_idx)
		# store copy of current valid indices for query sampling before optional updating with speaker information
		query_valid_idx = valid_idx.copy()
		# get indices of valid speakers according to speaker mode and cut valid indices
		if speakers is not None:
			if speaker_mode == "baseline":  # do nothing (i.e. no rules bro)
				pass
			elif speaker_mode == "easy":  # use same speaker throughout for the current class 
				unique_speakers_idx = np.where(np.equal(np.isin(
					unique_speakers, speaker_exclusion), False))[0]
				easy_speaker = unique_speakers[unique_speakers_idx][np.random.choice(len(unique_speakers_idx), size=1, replace=False)[0]]
				speaker_exclusion.append(easy_speaker)
				speaker_cls_idx = np.where(speakers == easy_speaker)[0]
				valid_idx = np.intersect1d(valid_idx, speaker_cls_idx)
			elif speaker_mode == "hard":  # query and support set speakers are disjoint
				valid_speakers_idx = np.where(np.equal(np.isin(
					speakers, speaker_exclusion), False))[0]
				valid_idx = np.intersect1d(valid_idx, valid_speakers_idx)
			elif speaker_mode == "distractor":  # use previous class episode speaker for support set "distractors"
				valid_speakers_idx = np.where(speakers == episode_speakers[cls_idx-1])[0]
				valid_idx = np.intersect1d(valid_idx, valid_speakers_idx)
			else:
				raise NotImplementedError("Speaker mode not implemented: {}".format(speaker_mode))
		# choose K support set indices
		support_cls_idx = np.random.choice(len(valid_idx), size=k_shot, replace=False)
		# exclude chosen support set indices from valid query indices
		query_valid_idx = np.setdiff1d(query_valid_idx, valid_idx[support_cls_idx])
		# choose size(query labels == class) query indices with optional speaker information
		if speakers is not None:
			if speaker_mode == "hard":  # query and support set speakers are disjoint
				query_speaker_exclusion.extend(speakers[valid_idx[support_cls_idx]])
				valid_speakers_idx = np.where(np.equal(np.isin(
					speakers, query_speaker_exclusion), False))[0]
				query_valid_idx = np.intersect1d(query_valid_idx, valid_speakers_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=num_cls_queries, replace=False)
				speaker_exclusion.extend(speakers[query_valid_idx[query_cls_idx]])
			elif speaker_mode == "distractor":  # use current class episode speaker for queries
				valid_speakers_idx = np.where(speakers == episode_speakers[cls_idx])[0]
				query_valid_idx = np.intersect1d(query_valid_idx, valid_speakers_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=num_cls_queries, replace=False)
			else:  # same valid indices as used for support set
				query_valid_idx = np.intersect1d(valid_idx, query_valid_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=num_cls_queries, replace=False)	
		else:
			query_cls_idx = np.random.choice(len(query_valid_idx), size=num_cls_queries, replace=False)
		# collect dataset indices selected for support and query sets
		support_idx.extend(valid_idx[support_cls_idx])
		query_idx.extend(query_valid_idx[query_cls_idx])
	# shuffle final support and query set indices if specified
	support_idx = np.asarray(support_idx)
	query_idx = np.asarray(query_idx)
	if shuffle:
		support_idx = np.random.permutation(support_idx)
		query_idx = np.random.permutation(query_idx)
	# double check that there are no duplicate uids in and between the support and query sets ...
	assert np.array_equal(sorted(uids[support_idx]), np.unique(uids[support_idx]))
	assert np.array_equal(sorted(uids[query_idx]), np.unique(uids[query_idx]))
	assert np.count_nonzero(np.isin(uids[query_idx], uids[support_idx])) == 0
	# return support and query set indices
	return support_idx, query_idx


def generate_episodes(uids, labels, speakers=None, n_episodes=600,
					  n_queries=15, k_shot=1, l_way=5, shuffle=True,
					  speaker_mode="baseline"):
	"""Generator function for sampling  K-shot L-way episode support and query set.
	
	See `sample_episode` for parameters.
	"""
	for _ in range(n_episodes):
		yield sample_episode(
			uids, labels, speakers=speakers,
			n_queries=n_queries, k_shot=k_shot, l_way=l_way, shuffle=shuffle,
			speaker_mode=speaker_mode)


def sample_multimodal_episode(paired_image_uids, paired_image_labels,
							  paired_speech_uids, paired_speech_labels,
							  image_to_speech_labels=None,
							  speakers=None,
							  n_queries=15, k_shot=1, l_way=5, shuffle=True,
							  strict_one_shot_matching=False,
							  speaker_mode="baseline"):
	"""Sample episode of K-shot L-way multimodal support set, N query speech, and L-way (1-shot) image matching set.
	
	See `sample_episode` for information on `speaker_mode`.
	"""
	# copy labels and uids and create shuffle indices (so that first instance of duplicate uids is different to previous episodes)
	assert len(paired_image_uids) == len(paired_speech_uids)  # loosely check that these are "paired"
	paired_image_uids = np.asarray(paired_image_uids).copy()
	paired_image_labels = np.asarray(paired_image_labels).copy()
	paired_speech_uids = np.asarray(paired_speech_uids).copy()
	paired_speech_labels = np.asarray(paired_speech_labels).copy()
	permute_idx = np.random.permutation(len(paired_image_uids))
	# get unique image class labels and check the image to speech label mapping
	image_classes = np.unique(paired_image_labels)
	if image_to_speech_labels is None:  # assume they are the same if not specified
		image_to_speech_labels = {label: label for label in image_classes}
	assert np.array_equal(sorted(image_to_speech_labels.keys()), sorted(image_classes))
	# get list of multimodal class choice pairs from the image to speech label mapping
	class_choices = []
	for image_cls, image_speech_cls in image_to_speech_labels.items():
		if type(image_speech_cls) is not list:
			image_speech_cls = [image_speech_cls]  # pretend its a list
		for speech_cls in image_speech_cls:
			class_choices.append((image_cls, speech_cls))
	class_choices = np.asarray(class_choices, dtype=np.object)
	# choose L unique paired class labels for this episode
	episode_cls_idx = np.random.choice(len(class_choices), size=l_way, replace=False)
	episode_cls = class_choices[episode_cls_idx]
	# choose N query labels from the chosen episode paired labels with replacement (uniform distribution among the N label pairs)
	query_cls_idx = np.random.choice(len(episode_cls), size=n_queries, replace=True)
	query_cls = episode_cls[query_cls_idx]
	# choose exactly one matching label for each of the chosen episode paired labels
	matching_cls = episode_cls
	if strict_one_shot_matching:  # remove duplicate image labels among the matching label pairs
		matching_unique_cls_idx = np.unique(matching_cls[:, 0], return_index=True)[1]
		matching_cls = matching_cls[matching_unique_cls_idx]
	# copy speakers and get unique speaker ids if speakers specified
	if speakers is not None:
		speakers = np.asarray(speakers).copy()
		unique_speakers = np.unique(speakers)
		# sample L unique speaker uids for the episode classes if using "distractor" speaker mode
		if speaker_mode == "distractor":
			episode_speakers = unique_speakers[np.random.choice(len(unique_speakers), size=l_way, replace=False)]
	# sample support, query and matching set
	support_paired_idx, query_speech_idx, matching_image_idx = [], [], []
	speaker_exclusion, query_speaker_exclusion = [], []
	for cls_idx, (image_cls, speech_cls) in enumerate(episode_cls):
		# set initial valid indices from which to sample as the shuffled indices
		valid_idx = permute_idx
		# get indices where paired dataset labels match the current paired class labels and cut valid indices
		image_label_cls_idx = np.where(paired_image_labels == image_cls)[0]
		speech_label_cls_idx = np.where(paired_speech_labels == speech_cls)[0]
		paired_label_cls_idx = np.intersect1d(image_label_cls_idx, speech_label_cls_idx)
		valid_idx = np.intersect1d(valid_idx, paired_label_cls_idx)
		# get number of paired query labels which match the current paired class labels
		image_query_cls_idx = np.where(query_cls[:, 0] == image_cls)[0]
		speech_query_cls_idx = np.where(query_cls[:, 1] == speech_cls)[0]
		paired_query_cls_idx = np.intersect1d(image_query_cls_idx, speech_query_cls_idx)
		num_cls_queries = len(paired_query_cls_idx)
		# get number of paired matching labels which match the current paired class labels
		image_match_cls_idx = np.where(matching_cls[:, 0] == image_cls)[0]
		speech_match_cls_idx = np.where(matching_cls[:, 1] == speech_cls)[0]
		paired_match_cls_idx = np.intersect1d(image_match_cls_idx, speech_match_cls_idx)
		num_cls_matching = len(paired_match_cls_idx)
		# get indices of unique paired uids (removing possible repetitions) and cut valid indices
		image_uids_unique_cls_idx = np.unique(paired_image_uids, return_index=True)[1]
		speech_uids_unique_cls_idx = np.unique(paired_speech_uids, return_index=True)[1]
		paired_uids_unique_cls_idx = np.intersect1d(image_uids_unique_cls_idx, speech_uids_unique_cls_idx)
		valid_idx = np.intersect1d(valid_idx, paired_uids_unique_cls_idx)
		# store copy of current valid indices for query sampling before optional updating with speaker information
		query_valid_idx = valid_idx.copy()
		# get indices of valid speakers according to speaker mode and cut valid indices
		if speakers is not None:
			if speaker_mode == "baseline":  # do nothing (i.e. no rules bro)
				pass
			elif speaker_mode == "easy":  # use same speaker throughout for the current class 
				unique_speakers_idx = np.where(np.equal(np.isin(
					unique_speakers, speaker_exclusion), False))[0]
				easy_speaker = unique_speakers[unique_speakers_idx][np.random.choice(len(unique_speakers_idx), size=1, replace=False)[0]]
				speaker_exclusion.append(easy_speaker)
				speaker_cls_idx = np.where(speakers == easy_speaker)[0]
				valid_idx = np.intersect1d(valid_idx, speaker_cls_idx)
			elif speaker_mode == "hard":  # query and support set speakers are disjoint
				valid_speakers_idx = np.where(np.equal(np.isin(
					speakers, speaker_exclusion), False))[0]
				valid_idx = np.intersect1d(valid_idx, valid_speakers_idx)
			elif speaker_mode == "distractor":  # use previous class episode speaker for support set "distractors"
				valid_speakers_idx = np.where(speakers == episode_speakers[cls_idx-1])[0]
				valid_idx = np.intersect1d(valid_idx, valid_speakers_idx)
			else:
				raise NotImplementedError("Speaker mode not implemented: {}".format(speaker_mode))
		# choose K support set indices
		support_cls_idx = np.random.choice(len(valid_idx), size=k_shot, replace=False)
		# exclude chosen support set indices from valid query indices
		query_valid_idx = np.setdiff1d(query_valid_idx, valid_idx[support_cls_idx])
		# choose size(query labels == class) query indices with optional speaker information
		if speakers is not None:
			if speaker_mode == "hard":  # query and support set speakers are disjoint
				query_speaker_exclusion.extend(speakers[valid_idx[support_cls_idx]])
				valid_speakers_idx = np.where(np.equal(np.isin(
					speakers, query_speaker_exclusion), False))[0]
				query_valid_idx = np.intersect1d(query_valid_idx, valid_speakers_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=(num_cls_queries + num_cls_matching), replace=False)
				speaker_exclusion.extend(speakers[query_valid_idx[query_cls_idx]])
			elif speaker_mode == "distractor":  # use current class episode speaker for queries
				valid_speakers_idx = np.where(speakers == episode_speakers[cls_idx])[0]
				query_valid_idx = np.intersect1d(query_valid_idx, valid_speakers_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=(num_cls_queries + num_cls_matching), replace=False)
			else:  # same valid indices as used for support set
				query_valid_idx = np.intersect1d(valid_idx, query_valid_idx)
				query_cls_idx = np.random.choice(len(query_valid_idx), size=(num_cls_queries + num_cls_matching), replace=False)	
		else:
			query_cls_idx = np.random.choice(len(query_valid_idx), size=(num_cls_queries + num_cls_matching), replace=False)
		# collect dataset indices selected for support, query and matching sets
		support_paired_idx.extend(valid_idx[support_cls_idx])
		if num_cls_matching > 0:  # add single matching index
			query_speech_idx.extend(query_valid_idx[query_cls_idx[:-1]])
			matching_image_idx.extend(query_valid_idx[query_cls_idx[-1:]])
		else:  # no matching index (duplicate label removed for strict one-shot matching)
			query_speech_idx.extend(query_valid_idx[query_cls_idx])
	# shuffle final support, query and matching set indices if specified
	support_paired_idx = np.asarray(support_paired_idx)
	query_speech_idx = np.asarray(query_speech_idx)
	matching_image_idx = np.asarray(matching_image_idx)
	if shuffle:
		support_paired_idx = np.random.permutation(support_paired_idx)
		query_speech_idx = np.random.permutation(query_speech_idx)
		matching_image_idx = np.random.permutation(matching_image_idx)
	# double check that there are no duplicate uids in and between the support and query sets ...
	assert np.array_equal(sorted(paired_image_uids[support_paired_idx]), np.unique(paired_image_uids[support_paired_idx]))
	assert np.array_equal(sorted(paired_speech_uids[support_paired_idx]), np.unique(paired_speech_uids[support_paired_idx]))
	assert np.array_equal(sorted(paired_image_uids[query_speech_idx]), np.unique(paired_image_uids[query_speech_idx]))
	assert np.array_equal(sorted(paired_speech_uids[matching_image_idx]), np.unique(paired_speech_uids[matching_image_idx]))
	assert np.count_nonzero(np.isin(paired_image_uids[query_speech_idx], paired_image_uids[support_paired_idx])) == 0
	assert np.count_nonzero(np.isin(paired_speech_uids[matching_image_idx], paired_speech_uids[support_paired_idx])) == 0
	# return paired support set, speech query set and image matching set indices
	return support_paired_idx, query_speech_idx, matching_image_idx


def generate_multimodal_episodes(paired_image_uids, paired_image_labels,
								 paired_speech_uids, paired_speech_labels,
								 image_to_speech_labels=None, speakers=None,
								 n_episodes=600, n_queries=15, k_shot=1, l_way=5,
								 shuffle=True, strict_one_shot_matching=False,
								 speaker_mode="baseline"):
	"""Generator function for sampling multimodal K-shot L-way episode support, query and matching set.
	
	See `sample_multimodal_episode` for parameters.
	"""
	for _ in range(n_episodes):
		yield sample_multimodal_episode(
			paired_image_uids, paired_image_labels,
			paired_speech_uids, paired_speech_labels,
			image_to_speech_labels=image_to_speech_labels, speakers=speakers,
			n_queries=n_queries, k_shot=k_shot, l_way=l_way, shuffle=shuffle,
			strict_one_shot_matching=strict_one_shot_matching,
			speaker_mode=speaker_mode)

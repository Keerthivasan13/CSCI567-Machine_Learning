import numpy as np
from util import accuracy
from hmm import HMM
from collections import Counter

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	S = len(tags)

	pi = np.zeros([S])
	A = np.full([S, S], 1/len(train_data))
	obs_dict, state_dict, idx = {}, {}, 0

	for line in train_data:
		for word in line.words:
			if word not in obs_dict:
				obs_dict[word] = idx
				idx += 1

	L = idx
	B = np.zeros([S, L])

	tag_cnt = Counter([line.tags[0] for line in train_data])

	for idx, tag in enumerate(tags):
		state_dict[tag] = idx
		pi[idx] = np.divide(tag_cnt[tag], L)

	for line in train_data:
		for idx in range(line.length - 1):
			A[state_dict[line.tags[idx]], state_dict[line.tags[idx+1]]] += 1

		for idx in range(line.length):
			B[state_dict[line.tags[idx]], obs_dict[line.words[idx]]] += 1

	for s in range(S):
		A[s] = np.divide(A[s], np.sum(A[s]))
		B[s] = np.divide(B[s], np.sum(B[s]))

	model = HMM(pi, A, B, obs_dict, state_dict)

	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	for line in test_data:
		cnt = 0
		for word in line.words:
			if word not in model.obs_dict:
				model.obs_dict[word] = len(model.obs_dict)
				cnt += 1
		if cnt:
			model.B = np.hstack([model.B, np.full([len(tags), cnt], 0.000001)])
		tagging.append(model.viterbi(line.words))
	###################################################
	return tagging

import pybloomfilter

import in1054.constants as consts

def povoate_bloom_filter(normal_dataframe, bf_filename=None):
	# concatenated_features_df = pd.read_csv(input_filename)
	concatenated_features_df = normal_dataframe.copy()
	concatenated_features_array = concatenated_features_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values

	# TODO: Update filter capacity based on the amount of data used to create
	# normal state representation
	fs_bloomfilter = pybloomfilter.BloomFilter(capacity=1000000, error_rate=0.01, filename=bf_filename)

	for feature in concatenated_features_array:
		fs_bloomfilter.add(feature.encode())

	return fs_bloomfilter


def check_bloomfilter(eval_dataframe, bloomfilter):
	tmp_df = eval_dataframe.copy()

	words_array = tmp_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values
	words_array = words_array.tolist()

	bf_results = []

	for word in words_array:
		single_result = word.encode() in bloomfilter
		bf_results.append(single_result)

	# TODO: Move these manipulations to other functions
	# Logical inversion of results
	# 0 means that is not in bloom filter
	# 1 means that it is in bloom filter
	inverted_bf_results = [not elem for elem in bf_results]
	# Now:
	# 0 means that is a regular frame
	# 1 means that is an injected frame

	# Convert boolean list to int list
	integer_map = map(int, inverted_bf_results)
	bf_int_results = list(integer_map)

	return bf_int_results

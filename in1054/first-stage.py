import pybloomfilter
import pandas as pd

def povoate_bloom_filter(input_filename):
	concatenated_features_df = pd.read_csv(input_filename)
	concatenated_features_array = concatenated_features_df.loc[:, "concatenated_features"].values

	# TODO: Update filter capacity based on the amount of data used to create
	# normal state representation
	fs_bloomfilter = pybloomfilter.BloomFilter(capacity=1000000, error_rate=0.01)

	for feature in concatenated_features_array:
		fs_bloomfilter.add(feature.encode())

	return fs_bloomfilter

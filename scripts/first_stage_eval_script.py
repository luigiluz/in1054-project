import pandas as pd
import pickle
import numpy as np
import tensorflow as tf

import in1054.constants as consts
import in1054.preprocessing as preprocessing
import in1054.first_stage as first_stage
import in1054.parser as parser
import in1054.metrics as metrics
import in1054.utils as utils
import in1054.second_stage as second_stage

data_folder_dir = consts.ROOT_PATH + "/data/"
kind_of_data_prefix = "RAND_nokf_noPCA"
train_filename = data_folder_dir + kind_of_data_prefix + "_train_DoS_dataset.csv"
validation_filename = data_folder_dir + kind_of_data_prefix + "_validation_DoS_dataset.csv"

def main():
	# Load train dataset
	train_df = pd.read_csv(train_filename)
	train_df = train_df.drop(columns="index")

	# Load validation dataset
	# Used for finding optimal hyper-parameters
	validation_df = pd.read_csv(validation_filename)
	validation_df = validation_df.drop(columns=["index", "level_0"])

	# Initiliaze training
	training_data = train_df
	validation_data = validation_df

	random_seed = 1
	# Setting random seeds
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)

	training_data_features, training_data_labels = utils.split_features_and_labels(training_data)
	validation_data_features, validation_data_labels = utils.split_features_and_labels(validation_data)

	train_features_np = second_stage.convert_df_to_numpy_array(training_data_features)
	validation_features_np = second_stage.convert_df_to_numpy_array(validation_data_features)

	training_data_labels_np = training_data_labels.to_numpy()
	validation_data_labels_np = validation_data_labels.to_numpy()

	# If using text vectorizer, this needs to be changed
	binary_train_features = train_features_np
	binary_validation_features = validation_features_np

	# Juntar os dados numa coluna unica
	converted_array = preprocessing.convert_to_comma_separated_string(binary_train_features)
	concatenated_train_features_df = pd.DataFrame(converted_array, columns = [consts.CONCATENATED_FEATURES_COLUMN_NAME])

	converted_val_array = preprocessing.convert_to_comma_separated_string(binary_validation_features)
	concatenated_validation_features_df = pd.DataFrame(converted_val_array, columns = [consts.CONCATENATED_FEATURES_COLUMN_NAME])

	# povoate bloom filter
	# print("povoating bloom filter")
	# fs_bloomfilter = first_stage.povoate_pybloom_filter(concatenated_train_features_df)

	filename = "bloom_filter.pickle"

	# # Save bloom filter as a pickle model
	# with open(filename ,'wb') as file:
	# 	pickle.dump(fs_bloomfilter, file)

	infile = open(filename, 'rb')
	loaded_bloomfilter = pickle.load(infile)
	infile.close()

	print(loaded_bloomfilter)

	# # get bloom filter results
	bf_results = first_stage.check_bloomfilter(concatenated_validation_features_df, loaded_bloomfilter)

	# evaluate bloom filter results
	conf_matrix = metrics.conf_matrix(validation_data_labels_np, bf_results)
	print("confusion matrix")
	print(conf_matrix)

	metrics_df = metrics.classification_metrics(validation_data_labels_np, bf_results)
	print(metrics_df)


if __name__ == "__main__":
	main()
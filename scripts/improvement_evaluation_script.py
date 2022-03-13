import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

import in1054.constants as consts
import in1054.utils as utils
import in1054.preprocessing as preprocessing
import in1054.metrics as metrics
import in1054.first_stage as first_stage

DATA_FOLDER = "/data/"
DATASET_POSTFIX = "_dataset.csv"

# Description parameters
DATASET_PREFIX = "DoS" # Change this according to the dataset used for experiments
METHOD = "DICT_BASED_ANOMALY"

DATASET_FILEPATH = consts.ROOT_PATH + DATA_FOLDER + DATASET_PREFIX + DATASET_POSTFIX

###### Helper functions ######
def flatten(t):
    return [item for sublist in t for item in sublist]


def join_data_bytes(array):
	n_rows = array.shape[0]
	converted_array = []

	for line in range(0, n_rows):
		print(str(line) + "/" + str(n_rows))
		converted_list = [str(element) for element in array[line, :]]
		joined_string = "".join(converted_list)
		converted_array.append([joined_string])

	return np.array(converted_array)


def convert_df_to_id_data_array(dataframe):
	tmp_df = dataframe.copy()
	#tmp_df = tmp_df.drop(["timestamp", "dlc", "flag"], axis=1)

	ids_array = tmp_df[["id"]].to_numpy()
	data_array = tmp_df.drop(["id"], axis=1).to_numpy()

	joined_data_array = join_data_bytes(data_array)

	ids_data_array = np.concatenate((ids_array, joined_data_array), axis=1)

	return ids_data_array

####### Main function ######
def main():
	# Define seeds to avoid randomized behavior
	random_seed = 1
	# Setting random seeds
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)

	# Dataset preprocessing steps
	input_df = pd.read_csv(DATASET_FILEPATH, names=consts.COLUMNS_NAMES)
	prepared_df = utils.prepare_test_dataset(input_df)
	preprocessed_df = preprocessing.convert_results_to_int(prepared_df)

	#preprocessed_features_df = preprocessing.preprocess(features_df)

	#preprocessed_entire_df = pd.concat([preprocessed_features_df, labels_df], axis=1)
	#preprocessed_entire_df.columns = consts.FEATURES_AND_LABEL_COLUMNS

	#print("preprocessed_entire_df")
	#print(preprocessed_entire_df.head(10))

	# Vou precisar modificar esse cara aqui
	# O test_df precisa ser o mesmo para todos os casos
	# O validation_df vai deixar de existir
	# O train_df vai variar de tamanho

	# Vou ter 3 casos de experimentos:
	# (i) Dict based bloom filter anomaly based
	# (ii) Dict based bloom filter signatura based
	# (iii) Hybrid method (signature based + anomaly based

	# No caso (i), os dados de treino sao dados normais
	# Nos casos (ii) e (iii), os dados de treino são ataques

	# O conjunto de teste vai ser sempre stratificado (mesma quantidade de dados normais e ataques)
	train_df, test_df = utils.random_train_test_split(preprocessed_df)

	train_data_features, _ = utils.split_features_and_labels(train_df)
	train_data_features = train_data_features.drop(columns=['index'])
	print("train_data_features")
	print(train_data_features.head(10))
	print(train_data_features.columns)

	test_data_features, test_data_labels = utils.split_features_and_labels(test_df)
	test_data_features = test_data_features.drop(columns=['index'])

	print("test_data_features")
	print(test_data_features.head(10))
	print(test_data_features.columns)

	print("test_data_labels")
	print(test_data_labels.head(10))

	# # Only necessary for test data, since all train data has the same label
	test_data_labels_np = test_data_labels.to_numpy()
	flatten_test_data_labels = flatten(test_data_labels_np)

	train_data_merged = convert_df_to_id_data_array(train_data_features)
	print("train_data_merged")
	print(train_data_merged)
	test_data_merged = convert_df_to_id_data_array(test_data_features)
	print("test_data_merged")
	print(test_data_merged)

	n_of_train_data_rows = train_data_features.shape[0]

	train_data_proportions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	metrics_list = []
	# Experiment loop
	for proportion in train_data_proportions:
		print("Current proportion = {}".format(proportion))
		proportional_n_of_rows = int(n_of_train_data_rows * proportion)
		proportional_train_data = train_data_merged[0:proportional_n_of_rows, :]

		# Train filter
		bf_dict = first_stage.BloomFilterDict()
		bf_dict.povoate_dict(proportional_train_data)

		# Load filter
		# filename = "DoS_dataset" + "bloom_filter.pickle"
		# infile = open(filename, 'rb')
		# bf_dict = pickle.load(infile)
		# infile.close()

		predictions = bf_dict.multiple_query(test_data_merged)
		counts = np.unique(predictions)
		print("prediction counts = " + str(counts))

		conf_matrix = flatten(metrics.conf_matrix(flatten_test_data_labels, predictions))
		classification_metrics = metrics.classification_metrics_as_array(flatten_test_data_labels, predictions)
		detection_time = [metrics.bloom_filter_detection_time(test_data_merged, bf_dict)]
		memory_consumption = [bf_dict.get_memory_consumption()]

		concatenated_metrics = [METHOD, DATASET_PREFIX, proportional_n_of_rows, *conf_matrix, *classification_metrics[0], *detection_time, *memory_consumption]

		metrics_list.append(concatenated_metrics)

	METRICS_COLUMNS = [
		"Method",
		"Dataset",
		"Training rows",
		"TN",
		"FP",
		"FN",
		"TP",
		"Accuracy",
		"Precision",
		"Recall",
		"F1-score",
		"Detection time",
		"Memory consumption"
	]

	metrics_df = pd.DataFrame(metrics_list, columns=METRICS_COLUMNS)
	print(metrics_df)
	metrics_df.to_csv("stratified_metrics_dataframe.csv")

if __name__ == "__main__":
	main()
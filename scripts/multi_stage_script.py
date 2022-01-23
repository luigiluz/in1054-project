import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

import in1054.constants as consts
import in1054.utils as utils
import in1054.first_stage as first_stage
import in1054.second_stage as second_stage
import in1054.preprocessing as preprocessing
import in1054.metrics as metrics

bloomfilter_filename = "bloom_filter.pickle"
bestmodel_filename = "best_dl_model.h5"

data_folder_dir = consts.ROOT_PATH + "/data/"
kind_of_data_prefix = "RAND_nokf_noPCA"
train_filename = data_folder_dir + kind_of_data_prefix + "_train_DoS_dataset.csv"
validation_filename = data_folder_dir + kind_of_data_prefix + "_validation_DoS_dataset.csv"
test_filename = data_folder_dir + kind_of_data_prefix + "_test_DoS_dataset.csv"

def main():
	# Fase de carregamento de dados

	# Load test dataset
	test_df = pd.read_csv(test_filename)
	test_df = test_df.drop(columns=["index"])
	print("test_df")
	print(test_df.head(10))

	# Prepare dataset
	# Setting random seeds
	random_seed = 1
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)

	test_data = test_df

	test_data_features, test_data_labels = utils.split_features_and_labels(test_data)

	test_features_np = second_stage.convert_df_to_numpy_array(test_data_features)
	test_data_labels_np = test_data_labels.to_numpy()

	# If using text vectorizer, this needs to be changed
	binary_test_features = test_features_np

	# Features converted for bloom filter
	converted_test_array = preprocessing.convert_to_comma_separated_string(binary_test_features)
	concatenated_test_features_df = pd.DataFrame(converted_test_array, columns = [consts.CONCATENATED_FEATURES_COLUMN_NAME])

	# Features converted for LSTM
	binary_test_features = utils.reshape_based_on_timestep(binary_test_features, n_of_timesteps=1)

	input_layer_shape = binary_test_features.shape[1:]

	# Fase de carregamento de modelo
	infile = open(bloomfilter_filename, 'rb')
	loaded_bloomfilter = pickle.load(infile)
	infile.close()

	# TODO: Iniciar o modelo de deep learning
	model_hyperparameters = {
		"n_of_dense_neurons": [8,4],
		"n_of_lstm_blocks": [4, 4],
		"overfit_avoidance" : {
			"dropout_rate" : 0.0,
			"regularizer_rate" : 0.0001,
			"max_norm": 3
		},
		"optimizer": {
			"learning_rate" : 0.00001,
			"momentum" : 0.0
		}
	}

	second_stage_model = second_stage.create_second_stage_model(input_layer_shape, model_hyperparameters)
	print(second_stage_model.summary())

	second_stage_model.load_weights(bestmodel_filename)
	print("modelo carregado")

	n_of_test_rows = len(test_data_labels_np)

	bf_results = first_stage.check_pybloomfilter(concatenated_test_features_df, loaded_bloomfilter)
	#print("bf_results")
	#print(bf_results)
	print("bf_results.unique")
	print(np.unique(bf_results))

	dl_results = second_stage.second_stage_model_predict(second_stage_model, binary_test_features)
	#print("dl_results")
	#print(dl_results)
	print("dl_results.unique")
	print(np.unique(dl_results))

	results_list = []
	for row in range(0, n_of_test_rows):
		current_result = dl_results[row]

		if (bf_results[row]) == 1:
			current_result = bf_results[row]

		results_list.append([int(current_result)])

	print("results_list")
	print(np.array(results_list, dtype=np.int64))
	print("results_list.unique")
	print(np.unique(results_list))

	print("test_data_labels_np")
	print(test_data_labels_np)
	print("test_data_labels_np.unique")
	print(np.unique(test_data_labels_np))

	metrics_df = metrics.classification_metrics(test_data_labels_np, results_list)
	print(metrics_df)

if __name__ == "__main__":
	main()

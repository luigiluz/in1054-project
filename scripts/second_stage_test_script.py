import pandas as pd
import tensorflow as tf
import numpy as np
import os

import in1054.constants as consts
import in1054.second_stage as second_stage
import in1054.metrics as metrics
import in1054.utils as utils

# TODO: Put filepaths in constants file
data_folder_dir = consts.ROOT_PATH + "/data/"
kind_of_data_prefix = "RAND_nokf_noPCA"
train_filename = data_folder_dir + kind_of_data_prefix + "_train_DoS_dataset.csv"
validation_filename = data_folder_dir + kind_of_data_prefix + "_validation_DoS_dataset.csv"
test_filename = data_folder_dir + kind_of_data_prefix + "_test_DoS_dataset.csv"

vectorizer_path = consts.ROOT_PATH + "/data/text_vectorizer"
model_path = consts.ROOT_PATH + '/data/ss_model_trained'

last_iter = 0

def get_model_name(k):
	return 'model_'+str(k)+'.h5'

def main():
	tf.keras.backend.clear_session()
	######################### Load data phase #########################
	# Load train dataset
	train_df = pd.read_csv(train_filename)
	train_df = train_df.drop(columns="index")
	print("train_df.info")
	print(train_df.info())

	# Load validation dataset
	# Used for finding optimal hyper-parameters
	validation_df = pd.read_csv(validation_filename)
	validation_df = validation_df.drop(columns=["index", "level_0"])
	print("validation-df.info")
	print(validation_df.info())

	# Merge train dataset and validation dataset (normal data only)
	normal_only_validation_df = validation_df[validation_df["flag"] == 0]

	train_df = pd.concat([train_df, normal_only_validation_df])
	train_df = train_df.reset_index()
	train_df = train_df.drop(columns="index")
	print("new train_df.info")
	print(train_df.info())
	print(train_df.head(10))

	# Load test dataset
	test_df = pd.read_csv(test_filename)#, names=consts.FEATURES_AND_LABEL_COLUMNS)
	test_df = test_df.drop(columns=["index"])
	print("test_df.info()")
	print(test_df.info())
	print(test_df.head(10))

	# TODO: Merge train and validation dataset (using only normal data)
	# Leave the rest to the testing set

	######################### Model preparing phase #########################
	# Load text vectorization layer
	loaded_vectorizer = None
	with tf.keras.utils.custom_object_scope({'custom_standardization': second_stage.custom_standardization}):
		loaded_vectorizer = second_stage.load_text_vectorizer(vectorizer_path)

	# Model hyperparameters
	# Best model hyperparameters based on
	random_seed = 1

	my_batch_size = 128
	my_epochs = 50

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

	######################### Directories ######################################
	saved_models_dir = consts.ROOT_PATH + "/test_saved_models/"

	my_prefix = "SEED{}DT{}BS{}EP{}NDN{},{}NLB{},{}LR{}MM{}DPR{}RR{}".format(
												random_seed,
												kind_of_data_prefix,
												my_batch_size,
												my_epochs,
												model_hyperparameters["n_of_dense_neurons"][0],
												model_hyperparameters["n_of_dense_neurons"][1],
												model_hyperparameters["n_of_lstm_blocks"][0],
												model_hyperparameters["n_of_lstm_blocks"][1],
												model_hyperparameters["optimizer"]["learning_rate"],
												model_hyperparameters["optimizer"]["momentum"],
												model_hyperparameters["overfit_avoidance"]["dropout_rate"],
												model_hyperparameters["overfit_avoidance"]["regularizer_rate"])
	my_model_folder = saved_models_dir + my_prefix + "model/"

	# try:
	# 	os.mkdir(my_model_folder)
	# except OSError as error:
	# 	global last_iter
	# 	last_iter += 1
	# 	my_model_folder = saved_models_dir + my_prefix + "model" + str(last_iter) + "/"
	# 	os.mkdir(my_model_folder)
	# 	print(error)

	my_logger = my_model_folder + "model_log.csv"
	fold_var = 1

	# Load second stage model
	# Setting random seeds
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)

	# Initiliaze training
	training_data = train_df
	test_data = test_df

	# Preprocessing phase
	# Convert data to numpy array
	# Change this split to concatenated split if using concatenated features
	training_data_features, training_data_labels = utils.split_features_and_labels(training_data)
	test_data_features, test_data_labels = utils.split_features_and_labels(test_data)

	train_features_np = second_stage.convert_df_to_numpy_array(training_data_features)
	test_features_np = second_stage.convert_df_to_numpy_array(test_data_features)

	training_data_labels_np = training_data_labels.to_numpy()
	test_data_labels_np = test_data_labels.to_numpy()

	# If using text vectorizer, this needs to be changed
	binary_train_features = train_features_np
	binary_test_features = test_features_np

	binary_train_features = utils.reshape_based_on_timestep(binary_train_features, n_of_timesteps=1)
	binary_test_features = utils.reshape_based_on_timestep(binary_test_features, n_of_timesteps=1)

	input_layer_shape = binary_train_features.shape[1:]

	# Create second stage model
	second_stage_model = second_stage.create_second_stage_model(input_layer_shape, model_hyperparameters)
	print(second_stage_model.summary())

	# ## Create callbacks
	# checkpoint = tf.keras.callbacks.ModelCheckpoint(my_model_folder + get_model_name(fold_var),
	# 												monitor='val_binary_accuracy', verbose=1,
	# 												save_best_only=True, mode='max')
	# csv_logger = tf.keras.callbacks.CSVLogger(my_logger,
	# 											append=True,
	# 											separator=',')
	# early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
	# 													patience=4,
	# 													verbose=1)
	# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
	# 													factor=0.1,
	# 													patiente=2,
	# 													min_delta=0,
	# 													min_lr=0.001 * model_hyperparameters["optimizer"]["learning_rate"])

	# callbacks_list = [
	# 					early_stopping,
	# 					checkpoint,
	# 					csv_logger,
	# 					reduce_lr
	# 				]

	# # Fit second stage model
	# second_stage_model.fit(binary_train_features,
	# 						training_data_labels_np,
	# 						batch_size=my_batch_size,
	# 						epochs=my_epochs,
	# 						callbacks=callbacks_list,
	# 						validation_data=(binary_test_features, test_data_labels_np))

	# Load best model to evaluate the performance of the model
	second_stage_model.load_weights(my_model_folder + get_model_name(fold_var))
	print("modelo carregado")

	tf.keras.backend.clear_session()

	## Prediction phase
	predictions = second_stage.second_stage_model_predict(second_stage_model, binary_test_features)

	print("predictions")
	print(predictions)

	print("test_data_labels")
	print(test_data_labels_np)

	## Evaluation phase
	# tive problema aqui
	metrics_df = metrics.classification_metrics(test_data_labels_np, predictions)
	print(metrics_df)


if __name__ == "__main__":
	main()
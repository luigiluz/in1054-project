import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import os

import in1054.second_stage as second_stage
import in1054.constants as consts
import in1054.utils as utils
# import in1054.metrics as metrics

# TODO: Put filepaths in constants file
data_folder_dir = consts.ROOT_PATH + "/data/"
kind_of_data_prefix = "RAND"
train_filename = data_folder_dir + kind_of_data_prefix + "_train_DoS_dataset.csv"
validation_filename = data_folder_dir + kind_of_data_prefix + "_validation_DoS_dataset.csv"
#test_filename = consts.ROOT_PATH + "/data/test_DoS_dataset.csv"
vectorizer_path = consts.ROOT_PATH + "/data/text_vectorizer"
model_output_path = consts.ROOT_PATH + '/data/ss_model_trained'

# Auxiliary function to get model name in each of the k iterations
def get_model_name(k):
	return 'model_'+str(k)+'.h5'

def main():
	####### Training phase ######

	## Load phase
	# Load train dataset
	# Used for training the detection algorithm
	train_df = pd.read_csv(train_filename)
	train_df = train_df.drop(columns="index")
	# Load validation dataset
	# Used for finding optimal hyper-parameters
	validation_df = pd.read_csv(validation_filename)
	validation_df = validation_df.drop(columns="index")

	## Model initialization phase
	# Load text vectorization layer
	loaded_vectorizer = None
	with tf.keras.utils.custom_object_scope({'custom_standardization': second_stage.custom_standardization}):
		loaded_vectorizer = second_stage.load_text_vectorizer(vectorizer_path)
	print("Vectorizer successfully loaded")

	## Training phase
	# Model parameteres
	my_batch_size = 1024
	my_epochs = 100

	model_hyperparameters = {
		"n_of_dense_neurons": 16,
		"n_of_lstm_blocks": 4,
		"overfit_avoidance" : {
			"dropout_rate" : 0.4,
			"regularizer_rate" : 0.0001,
			"max_norm": 3
		},
		"optimizer": {
			"learning_rate" : 0.00001,
			"momentum" : 0.0
		}
	}

	# Directiories
	saved_models_dir = consts.ROOT_PATH + "/saved_models/"
	# logger_folder_dir = saved_models_dir

	my_prefix = "DT{}BS{}EP{}NDN{}NLB{}LR{}MM{}".format(
												kind_of_data_prefix,
												my_batch_size,
												my_epochs,
												model_hyperparameters["n_of_dense_neurons"],
												model_hyperparameters["n_of_lstm_blocks"],
												model_hyperparameters["optimizer"]["learning_rate"],
												model_hyperparameters["optimizer"]["momentum"])
	my_model_folder = saved_models_dir + my_prefix + "model/"
	my_logger = saved_models_dir + my_prefix + "model_log.csv"
	# TODO: Adicionar alguma coisa para adicionar um nome nome
	# Talvez algum indice
	try:
		os.mkdir(my_model_folder)
	except OSError as error:
		print(error)

	# Lists to hold metrics
	VALIDATION_ACCURACY = []
	VALIDATION_LOSS = []

	# KFold iniatialization
	fold_var = 1
	kf = KFold(n_splits=10)

	# Initiliaze training
	for train_index, val_index in kf.split(train_df):
		print("Split number: " + str(fold_var))
		# Data split
		training_data = train_df.iloc[train_index]
		validation_data = train_df.iloc[val_index]
		validation_data = pd.concat([validation_data, validation_df])

		# Preprocessing phase
		# Convert data to numpy array
		# Change this split to concatenated split if using concatenated features
		training_data_features, training_data_labels = utils.split_features_and_labels(training_data)
		validation_data_features, validation_data_labels = utils.split_features_and_labels(validation_data)

		train_features_np = second_stage.convert_df_to_numpy_array(training_data_features)
		validation_features_np = second_stage.convert_df_to_numpy_array(validation_data_features)

		# If using text vectorizer, this needs to be changed
		binary_train_features = train_features_np
		binary_validation_features = validation_features_np

		binary_train_features = utils.reshape_based_on_timestep(binary_train_features, n_of_timesteps=1)
		binary_validation_features = utils.reshape_based_on_timestep(binary_validation_features, n_of_timesteps=1)

		# Create second stage model
		input_layer_shape = binary_train_features.shape[1:]
		second_stage_model = second_stage.create_second_stage_model(input_layer_shape, model_hyperparameters)
		print(second_stage_model.summary())

		# # Create callbacks
		checkpoint = tf.keras.callbacks.ModelCheckpoint(my_model_folder + get_model_name(fold_var),
														monitor='val_binary_accuracy', verbose=1,
														save_best_only=True, mode='max')
		csv_logger = tf.keras.callbacks.CSVLogger(my_logger,
													append=True,
													separator=',')
		early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

		callbacks_list = [
							early_stopping,
							checkpoint,
							csv_logger
						]

		# Fit second stage model
		second_stage_model.fit(binary_train_features,
								training_data_labels,
								batch_size=my_batch_size,
								epochs=my_epochs,
								callbacks=callbacks_list,
								validation_data=(binary_validation_features, validation_data_labels))

		# Load best model to evaluate the performance of the model
		second_stage_model.load_weights(my_model_folder + get_model_name(fold_var))

		results = second_stage_model.evaluate(x = binary_validation_features,
												y= validation_data_labels)
		results = dict(zip(second_stage_model.metrics_names,results))

		VALIDATION_ACCURACY.append(results['binary_accuracy'])
		VALIDATION_LOSS.append(results['loss'])

		tf.keras.backend.clear_session()

		fold_var += 1

	print("VALIDATION_ACCURACY")
	print(VALIDATION_ACCURACY)
	print("VALIDATION_LOSS")
	print(VALIDATION_LOSS)

	# # Save trained model
	# tf.keras.models.save_model(second_stage_model, model_output_path)
	# print("second stage model properly saved")


if __name__ == "__main__":
	main()
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import in1054.second_stage as second_stage
import in1054.constants as consts
import in1054.utils as utils
# import in1054.metrics as metrics

# TODO: Put filepaths in constants file
train_filename = consts.ROOT_PATH + "/data/Ftrain_DoS_dataset.csv"
validation_filename = consts.ROOT_PATH + "/data/Fvalidation_DoS_dataset.csv"
#test_filename = consts.ROOT_PATH + "/data/test_DoS_dataset.csv"
vectorizer_path = consts.ROOT_PATH + "/data/text_vectorizer"
model_output_path = consts.ROOT_PATH + '/data/ss_model_trained'

# Auxiliary function to get model name in each of the k iterations
def get_model_name(k):
	return 'model_'+str(k)+'.h5'

def convert_list(list):
	tmp_list = []

	for item in list:
		tmp_list.append([item])

	return np.array(tmp_list)

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
	print("######## Train dataset ########")
	print(train_df.head(10))
	print("######## Validation dataset ########")
	print(validation_df.head(10))

	## Model initialization phase
	# Load text vectorization layer
	loaded_vectorizer = None
	with tf.keras.utils.custom_object_scope({'custom_standardization': second_stage.custom_standardization}):
		loaded_vectorizer = second_stage.load_text_vectorizer(vectorizer_path)
	print("Vectorizer successfully loaded")

	# Training phase
	VALIDATION_ACCURACY = []
	VALIDATION_LOSS = []
	save_dir = consts.ROOT_PATH + "/data/saved_models/"
	fold_var = 1

	kf = KFold(n_splits=10)

	for train_index, val_index in kf.split(train_df):
		print("Split number: " + str(fold_var))
		# Data split
		training_data = train_df.iloc[train_index]
		validation_data = train_df.iloc[val_index]
		validation_data = pd.concat([validation_data, validation_df])

		# Preprocessing phase
		# Convert data to numpy array
		training_data_features, training_data_labels = utils.split_features_and_labels(training_data)#utils.split_concatenated_features_labels(training_data)
		validation_data_features, validation_data_labels = utils.split_features_and_labels(validation_data)#utils.split_concatenated_features_labels(validation_data)

		train_features_np = second_stage.convert_df_to_numpy_array(training_data_features)
		validation_features_np = second_stage.convert_df_to_numpy_array(validation_data_features)

		binary_train_features = train_features_np#loaded_vectorizer(train_features_np)
		binary_validation_features = validation_features_np#loaded_vectorizer(validation_features_np)

		training_data_labels = training_data_labels.to_numpy()
		validation_data_labels = validation_data_labels.to_numpy()

		print("************* shapes *****************")
		print("binary_train_features.shape")
		print(binary_train_features.shape)
		print("binary_validation_features.shape")
		print(binary_validation_features.shape)

		print("training_data_labels.shape")
		print(training_data_labels.shape)
		print("validation_data_labels.shape")
		print(validation_data_labels.shape)
		print("************* end of shapes *****************")

		# Manipulating data for LSTM
		binary_train_features = binary_train_features.reshape((binary_train_features.shape[0], 1, binary_train_features.shape[1]))
		binary_validation_features = binary_validation_features.reshape((binary_validation_features.shape[0], 1, binary_validation_features.shape[1]))

		# Create second stage model
		# It creates and compile the model
		input_layer_shape = binary_train_features.shape[1:]
		second_stage_model = second_stage.create_second_stage_model(input_layer_shape)
		print("second stage model")
		print(second_stage_model.summary())

		# # Create callbacks
		checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(fold_var),
														monitor='val_accuracy', verbose=1,
														save_best_only=True, mode='max')

		callbacks_list = [
							checkpoint
						]

		second_stage_model.fit(binary_train_features,
								training_data_labels,
								batch_size=128,
								epochs=100,
								callbacks=callbacks_list,
								validation_data=(binary_validation_features, validation_data_labels))

		# Load best model to evaluate the performance of the model
		second_stage_model.load_weights(consts.ROOT_PATH + "/data/saved_models/model_"+str(fold_var)+".h5")

		results = second_stage_model.evaluate(x = binary_validation_features,
												y= validation_data_labels)
		results = dict(zip(second_stage_model.metrics_names,results))

		VALIDATION_ACCURACY.append(results['accuracy'])
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
import pandas as pd
import numpy as np
import tensorflow as tf

import in1054.second_stage as second_stage
import in1054.constants as consts
# import in1054.utils as utils
# import in1054.metrics as metrics

# TODO: Put filepaths in constants file
train_filename = consts.ROOT_PATH + "/data/normal_run_data_joined.csv"
vectorizer_path = consts.ROOT_PATH + "/data/text_vectorizer"
model_output_path = consts.ROOT_PATH + '/data/ss_model_trained'

def main():
	####### Training phase ######

	## Load phase
	# Load train dataset
	train_df = pd.read_csv(train_filename)
	print(train_df.head(10))
	print("first line")
	print(train_df.iloc[0, 0])
	print("size of characters from the first line")
	print(len(train_df.iloc[0, 0]))

	## Model initialization phase
	# Load text vectorization layer
	loaded_vectorizer = None
	with tf.keras.utils.custom_object_scope({'custom_standardization': second_stage.custom_standardization}):
		loaded_vectorizer = second_stage.load_text_vectorizer(vectorizer_path)

	# Create second stage model
	second_stage_model = second_stage.create_second_stage_model()
	print("model created")

	## Preprocessing phase
	# Convert data to numpy array
	train_df_as_np = second_stage.convert_df_to_numpy_array(train_df)
	binary_data_array = loaded_vectorizer(train_df_as_np)
	print("train data successfully converted")
	print(binary_data_array[0].shape)

	# Create train data labels
	train_labels_array = np.zeros(binary_data_array.shape[0])
	print("labels array successfully created")

	# ## Training phase
	# second_stage_model = second_stage.train_second_stage_model(second_stage_model, binary_data_array, train_labels_array)
	# print("second stage model properly trained")

	# # Save trained model
	# tf.keras.models.save_model(second_stage_model, model_output_path)
	# print("second stage model properly saved")



if __name__ == "__main__":
	main()
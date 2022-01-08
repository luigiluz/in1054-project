import pandas as pd
import tensorflow as tf

import in1054.constants as consts
import in1054.second_stage as second_stage
import in1054.metrics as metrics
import in1054.utils as utils

# TODO: Put filepaths in constants file
test_filename = consts.ROOT_PATH + "/data/minDoS_dataset.csv"
vectorizer_path = consts.ROOT_PATH + "/data/text_vectorizer"
model_path = consts.ROOT_PATH + '/data/ss_model_trained'

def main():
	# Load test dataset
	test_df = pd.read_csv(test_filename, names=consts.COLUMNS_NAMES)
	print(test_df.head(10))
	print(test_df.columns)

	## Loading model phase
	# Load text vectorization layer
	loaded_vectorizer = None
	with tf.keras.utils.custom_object_scope({'custom_standardization': second_stage.custom_standardization}):
		loaded_vectorizer = second_stage.load_text_vectorizer(vectorizer_path)
	print("vetorizador carregado")

	# Load second stage model
	loaded_second_stage_model = tf.keras.models.load_model(model_path)
	print("modelo treinado carregado")
	print(loaded_second_stage_model.summary())

	## Preprocessing phase
	# Preprocess test dataset
	# TODO: Change to use method that splits considering the correct proportion
	# tá dando problema aqui
	features_df, prepared_labels = utils.features_labels_split(test_df)
	print("features_df")
	print(features_df.head(10))
	print("features_df.columns")
	print(features_df.columns)
	print("first line")
	print(features_df.iloc[0, 0])
	print("size of characters from the first line")
	print(len(features_df.iloc[0, 0]))

	features_df_as_np = second_stage.convert_df_to_numpy_array(features_df)
	binary_test_data = loaded_vectorizer(features_df_as_np)
	print("binary_test_data")
	print(binary_test_data.shape)
	print("binary_test_data[0]")
	print(binary_test_data[0])

	## Prediction phase
	predictions = second_stage.second_stage_model_predict(loaded_second_stage_model, binary_test_data)

	## Evaluation phase
	metrics_df = metrics.classification_metrics(prepared_labels, predictions)
	print(metrics_df)

if __name__ == "__main__":
	main()
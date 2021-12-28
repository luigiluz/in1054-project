import pandas as pd
import numpy as np

import in1054.second_stage as second_stage
import in1054.constants as consts
import in1054.utils as utils
import in1054.metrics as metrics

train_filename = consts.ROOT_PATH + "/data/normal_run_data_joined_50.csv"

def main():
	# Load train dataset
	train_df = pd.read_csv(train_filename)
	print(train_df.head(10))

	# Load test dataset
	test_df = pd.read_csv(consts.DOS_DATA_CSV_PATH, names=consts.COLUMNS_NAMES, index_col=0)
	print("test_df.columns")
	print(test_df.columns)
	prep_test_features, prep_test_labels = utils.split_test_dataset(test_df)
	print("test dataset loaded and prepared")

	# Train the second stage model
	binary_data_array = second_stage.prepare_input_data(train_df)
	train_labels_array = np.zeros(50) # TODO: Fazer isso de uma maneira esperta
	second_stage_model = second_stage.create_second_stage_model()
	second_stage_model = second_stage.train_second_stage_model(second_stage_model, binary_data_array, train_labels_array)
	print("second stage model properly trained")

	# Predict using the second stage model
	# Tenho que preparar as features que vao ser utilizadas pra testar
	binary_test_features = second_stage.prepare_input_data(prep_test_features)
	predictions = second_stage.second_stage_model_predict(second_stage_model, binary_test_features)
	print("predictions made")

	# Calculate the second stage model metrics
	metrics_df = metrics.classification_metrics(prep_test_labels, predictions)
	print(metrics_df)



if __name__ == "__main__":
	main()
import pandas as pd

import in1054.constants as consts
import in1054.second_stage as second_stage

train_filename = consts.ROOT_PATH + "/data/normal_run_data_joined.csv"
output_path = consts.ROOT_PATH + "/data/text_vectorizer"

def main():
	# Load dataset
	train_df = pd.read_csv(train_filename)
	print("Dataset successfully loaded")

	# Convert data to numpy array
	train_data_np = second_stage.convert_df_to_numpy_array(train_df)
	print("Data successfully converted")

	# Create and save text vectorizer
	# TODO: Which data is necessary to be at text vectorizer creation?
	# TODO: Should it be all data?
	second_stage.create_text_vectorization_layer(train_data_np, output_path)
	print("Text vectorizator properly saved")


if __name__ == "__main__":
	main()
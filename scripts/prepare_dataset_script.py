import in1054.constants as consts
import in1054.preprocessing as preprocessing
import in1054.utils as utils

import pandas as pd

data_folder = "/data/"
filename = "DoS_dataset.csv"

input_filepath = consts.ROOT_PATH + data_folder + filename
output_filepath = consts.ROOT_PATH + data_folder + "prepared_" + filename

def main():
	input_df = pd.read_csv(input_filepath, names=consts.COLUMNS_NAMES)
	print(input_df.head(10))

	prepared_df = utils.prepare_test_dataset(input_df)
	prepared_df = preprocessing.convert_results_to_int(prepared_df)
	# no preprocess, eu nao posso tentar converter as labels
	features_df, labels_df = utils.split_features_and_labels(prepared_df)
	preprocessed_features_df = preprocessing.preprocess(features_df)

	# concatenar as labels no preprocessed
	preprocessed_entire_df = pd.concat([preprocessed_features_df, labels_df], axis=1)
	train_df, val_injected_df, test_df = utils.train_validation_test_split(preprocessed_entire_df)
	# depois de separar, eu ainda vou precisar preprocessar as labels
	# ou seja, ainda preciso converter as labels para 0 e 1

	print("train_df.info()")
	print(train_df.info())
	print("val_injected_df.info()")
	print(val_injected_df.info())
	print("test_df.info()")
	print(test_df.info())

	train_df.to_csv(consts.ROOT_PATH + data_folder + "train_" + filename, index=False)
	val_injected_df.to_csv(consts.ROOT_PATH + data_folder + "validation_" + filename, index=False)
	test_df.to_csv(consts.ROOT_PATH + data_folder + "test_" + filename, index=False)

if __name__ == "__main__":
	main()
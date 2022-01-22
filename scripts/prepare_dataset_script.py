import in1054.constants as consts
import in1054.preprocessing as preprocessing
import in1054.utils as utils

import pandas as pd

data_folder = "/data/"
filename = "DoS_dataset.csv"

input_filepath = consts.ROOT_PATH + data_folder + filename
output_filepath = consts.ROOT_PATH + data_folder + "RAND_" + filename

def main():
	input_df = pd.read_csv(input_filepath, names=consts.COLUMNS_NAMES)
	print(input_df.head(10))

	prepared_df = utils.prepare_test_dataset(input_df)
	print("prepared_df")
	print(prepared_df.head(10))
	preprocessed_df = preprocessing.convert_results_to_int(prepared_df)
	print("preprocessed_df")
	print(preprocessed_df.head(10))
	# no preprocess, eu nao posso tentar converter as labels
	features_df, labels_df = utils.split_features_and_labels(preprocessed_df)

	print("features_df")
	print(features_df.head(10))

	print("labels_df")
	print(labels_df.head(10))

	preprocessed_features_df = preprocessing.preprocess(features_df)

	print("prepocessed_features_df")
	print(preprocessed_features_df)

	# print("labels_df")
	# print(labels_df)

	# concatenar as labels no preprocessed
	# o nome das colunas ta sendo alterado aqui
	preprocessed_entire_df = pd.concat([preprocessed_features_df, labels_df], axis=1)
	preprocessed_entire_df.columns = consts.FEATURES_AND_LABEL_COLUMNS

	print("preprocessed_entire_df")
	print(preprocessed_entire_df.head(10))

	train_df, validation_df, test_df = utils.random_train_validation_test_split(preprocessed_entire_df)
	# depois de separar, eu ainda vou precisar preprocessar as labels
	# ou seja, ainda preciso converter as labels para 0 e 1

	print("train_df.info()")
	print(train_df.info())
	print("validation_df.info()")
	print(validation_df.info())
	print("test_df.info()")
	print(test_df.info())

	train_df.to_csv(consts.ROOT_PATH + data_folder + "RAND_nokf_noPCA_train_" + filename, index=False)
	validation_df.to_csv(consts.ROOT_PATH + data_folder + "RAND_nokf_noPCA_validation_" + filename, index=False)
	test_df.to_csv(consts.ROOT_PATH + data_folder + "RAND_nokf_noPCA_test_" + filename, index=False)

if __name__ == "__main__":
	main()
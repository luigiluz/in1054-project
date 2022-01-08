import in1054.preprocessing as preprocessing
import in1054.constants as consts
import in1054.parser as parser

import pandas as pd

def _prepare_test_dataset(dataframe):
	tmp_df = dataframe.copy()

	tmp_df = preprocessing.filter_by_dlc(tmp_df, dlc=8)
	tmp_df = parser.convert_cols_from_hex_string_to_int(tmp_df, consts.COLUMNS_TO_CONVERT)

	return tmp_df


def _split_features_and_labels(dataframe):
	tmp_df = dataframe.copy()
	print(tmp_df.columns)

	labels_df = tmp_df[[consts.FLAG_COLUMN_NAME]]
	features_df = tmp_df.drop(columns=[consts.FLAG_COLUMN_NAME])

	return features_df, labels_df


def _prepare_features(features_dataframe):
	tmp_df = features_dataframe.copy()

	prepared_features_df = preprocessing.preprocess(tmp_df)

	return prepared_features_df


def _prepare_labels(labels_dataframe):
	tmp_df = labels_dataframe.copy()

	preprocessed_labels = preprocessing.convert_results_to_int(tmp_df)

	labels = preprocessed_labels.loc[:, consts.FLAG_COLUMN_NAME].tolist()

	return labels


def train_validation_test_split(dataframe):
	tmp_df = _prepare_test_dataset(dataframe)
	# aqui eu tenho todas as colunas ainda
	regular_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.REGULAR_FLAG_STR]
	injected_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.INJECTED_FLAG_STR]

	train_df = regular_df.sample(n=consts.NUMBER_OF_TRAINING_SAMPLES)
	train_df = train_df.reset_index()

	regular_df.drop(train_df.index, inplace=True)

	#val_regular_df = regular_df.sample(n=consts.NUMBER_OF_REGULAR_VALIDATION_SAMPLES)
	val_injected_df = regular_df.sample(n=consts.NUMBER_OF_INJECTED_VALIDATION_SAMPLES)
	val_injected_df = val_injected_df.reset_index()

	#regular_df.drop(val_regular_df.index, inplace=True)
	injected_df.drop(val_injected_df.index, inplace=True)

	#validation_df = pd.concat([val_regular_df, val_injected_df])
	#validation_df = validation_df.reset_index()
	test_df = pd.concat([regular_df, injected_df])
	test_df = test_df.reset_index()

	return train_df, val_injected_df, test_df


def features_labels_split(dataframe):
	tmp_df = dataframe.copy()

	tmp_df = preprocessing.filter_by_dlc(tmp_df, dlc=8)
	tmp_df = parser.convert_cols_from_hex_string_to_int(tmp_df, consts.COLUMNS_TO_CONVERT)

	features_df, labels_df = _split_features_and_labels(tmp_df)
	prepared_features = _prepare_features(features_df)
	prepared_labels = _prepare_labels(labels_df)

	return prepared_features, prepared_labels
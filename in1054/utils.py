import in1054.preprocessing as preprocessing
import in1054.constants as consts
import in1054.parser as parser

import pandas as pd

def prepare_test_dataset(dataframe):
	tmp_df = dataframe.copy()

	tmp_df = preprocessing.filter_by_dlc(tmp_df, dlc=8)
	tmp_df = parser.convert_cols_from_hex_string_to_int(tmp_df, consts.COLUMNS_TO_CONVERT)

	return tmp_df.drop(columns='index')


def split_features_and_labels(dataframe):
	tmp_df = dataframe.copy()
	#print(tmp_df.columns)

	labels_df = tmp_df[[consts.FLAG_COLUMN_NAME]]
	labels_df = labels_df.to_numpy()

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


def random_train_validation_test_split(dataframe):
	tmp_df = dataframe.copy()#_prepare_test_dataset(dataframe)
	# aqui eu tenho todas as colunas ainda
	regular_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.REGULAR_FLAG_INT]
	injected_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.INJECTED_FLAG_INT]

	train_df = regular_df.sample(n=consts.NUMBER_OF_TRAINING_SAMPLES)
	regular_df.drop(train_df.index, inplace=True)
	train_df = train_df.reset_index()

	#val_regular_df = regular_df.sample(n=consts.NUMBER_OF_REGULAR_VALIDATION_SAMPLES)
	val_injected_df = injected_df.sample(n=consts.NUMBER_OF_INJECTED_VALIDATION_SAMPLES)

	#regular_df.drop(val_regular_df.index, inplace=True)
	injected_df.drop(val_injected_df.index, inplace=True)
	val_injected_df = val_injected_df.reset_index()

	#validation_df = pd.concat([val_regular_df, val_injected_df])
	#validation_df = validation_df.reset_index()
	test_df = pd.concat([regular_df, injected_df])
	test_df = test_df.reset_index()

	return train_df, val_injected_df, test_df


def sequence_train_validation_test_split(dataframe):
	tmp_df = dataframe.copy()

	regular_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.REGULAR_FLAG_INT]
	injected_df = tmp_df[tmp_df[consts.FLAG_COLUMN_NAME] == consts.INJECTED_FLAG_INT]

	train_df = regular_df.iloc[0:consts.NUMBER_OF_TRAINING_SAMPLES, :]
	train_df = train_df.reset_index()

	val_injected_df = injected_df.iloc[0:consts.NUMBER_OF_INJECTED_VALIDATION_SAMPLES, :]
	val_injected_df = val_injected_df.reset_index()

	rem_train_df = regular_df.iloc[consts.NUMBER_OF_TRAINING_SAMPLES:-1, :]
	rem_injected_df = injected_df.iloc[consts.NUMBER_OF_REGULAR_VALIDATION_SAMPLES:-1, :]

	test_df = pd.concat([rem_train_df, rem_injected_df])
	test_df = test_df.reset_index()

	return train_df, val_injected_df, test_df


def features_labels_split(dataframe):
	tmp_df = dataframe.copy()

	tmp_df = preprocessing.filter_by_dlc(tmp_df, dlc=8)
	tmp_df = parser.convert_cols_from_hex_string_to_int(tmp_df, consts.COLUMNS_TO_CONVERT)

	features_df, labels_df = split_features_and_labels(tmp_df)
	prepared_features = _prepare_features(features_df)
	prepared_labels = _prepare_labels(labels_df)

	return prepared_features, prepared_labels


def split_concatenated_features_labels(dataframe):
	tmp_df = dataframe.copy()

	features_df = tmp_df.loc[:, "concatenated_features"]
	labels_df = tmp_df.loc[:, "flag"]

	return features_df, labels_df


def reshape_based_on_timestep(data_array, n_of_timesteps=1):
	n_of_rows = data_array.shape[0]
	n_of_cols = data_array.shape[1]

	n_of_samples = int(n_of_rows / n_of_timesteps)

	reshaped_data = data_array.reshape((n_of_samples, n_of_timesteps, n_of_cols))

	return reshaped_data
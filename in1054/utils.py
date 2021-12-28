import in1054.preprocessing as preprocessing
import in1054.constants as consts
import in1054.parser as parser

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


def split_test_dataset(dataframe):
	tmp_df = _prepare_test_dataset(dataframe)
	features_df, labels_df = _split_features_and_labels(tmp_df)
	prepared_features = _prepare_features(features_df)
	prepared_labels = _prepare_labels(labels_df)

	return prepared_features, prepared_labels
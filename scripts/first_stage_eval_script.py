import pandas as pd

import in1054.constants as consts
import in1054.preprocessing as preprocessing
import in1054.first_stage as first_stage
import in1054.parser as parser
import in1054.metrics as metrics

def main():
	# load normal dataframe
	normal_joined_df = pd.read_csv(consts.NORMAL_RUN_DATA_JOINED_PATH)

	# load malicious dataframe
	dos_df = pd.read_csv(consts.DOS_DATA_CSV_PATH, names=consts.COLUMNS_NAMES)
	# filter dlc columns
	dos_df = preprocessing.filter_by_dlc(dos_df, 8)
	print("dos_df")
	print(dos_df.head(10))

	dos_results_column_df = dos_df[[consts.FLAG_COLUMN_NAME]]
	print("dos_results_column_df")
	print(dos_results_column_df.head(10))

	dos_results_column_df = preprocessing.convert_results_to_int(dos_results_column_df)
	print("dos_results_column_df_after_conversion")
	print(dos_results_column_df.head(10))
	dos_results = dos_results_column_df.loc[:, consts.FLAG_COLUMN_NAME].tolist()

	# preprocess malicious dataframe
	dos_df = parser.convert_cols_from_hex_string_to_int(dos_df, consts.COLUMNS_TO_CONVERT)
	dos_df = preprocessing.preprocess(dos_df)
	print("preprocessed_df")
	print(dos_df)

	# povoate bloom filter
	print("povoating bloom filter")
	fs_bloomfilter = first_stage.povoate_bloom_filter(normal_joined_df)

	# get bloom filter results
	bf_results = first_stage.check_bloomfilter(dos_df, fs_bloomfilter)

	# evaluate bloom filter results
	conf_matrix = metrics.conf_matrix(dos_results, bf_results)
	print("confusion matrix")
	print(conf_matrix)

	metrics_df = metrics.classification_metrics(dos_results, bf_results)
	print(metrics_df)


if __name__ == "__main__":
	main()
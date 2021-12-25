import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

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
	# 0 means that is not in bloom filter
	# 1 means that it is in bloom filter
	# print("bloom filter results")
	# print(bf_results)

	# evaluate bloom filter results
	conf_matrix = metrics.conf_matrix(dos_results, bf_results)
	print("confusion matrix")
	print(conf_matrix)

	accuracy = accuracy_score(dos_results, bf_results)
	print("accuracy")
	print(accuracy)

	precision = precision_score(dos_results, bf_results)
	print("precision")
	print(precision)

	recall = recall_score(dos_results, bf_results)
	print("recall")
	print(recall)

	f1_score_metric = f1_score(dos_results, bf_results)
	print("f1 score")
	print(f1_score_metric)


if __name__ == "__main__":
	main()
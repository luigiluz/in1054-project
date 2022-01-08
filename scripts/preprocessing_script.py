import pandas as pd

import in1054.preprocessing
import in1054.constants as consts

filepath = consts.ROOT_PATH + "/data/DoS_dataset.csv"
output_filepath = consts.ROOT_PATH + "/data/joinedDoS_dataset.csv"

def main():
	features_df = pd.read_csv(filepath, names=consts.COLUMNS_NAMES)
	print(features_df.head(10))
	in1054.preprocessing.preprocess(features_df, output_filepath)

if __name__ == "__main__":
	main()
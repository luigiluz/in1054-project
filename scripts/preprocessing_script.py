import pandas as pd

import in1054.preprocessing
import in1054.constants as consts

def main():
	features_df = pd.read_csv(consts.NORMAL_RUN_DATA_CSV_PATH)
	print(features_df.head(10))
	preprocessed_features = in1054.preprocessing.preprocess(features_df)
	print(preprocessed_features.shape)

if __name__ == "__main__":
	main()
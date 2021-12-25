import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import in1054.constants as consts

def filter_by_dlc(dataframe, dlc):
  tmp_df = dataframe.copy()
  tmp_df = tmp_df[tmp_df[consts.DLC_COLUMN_NAME] == dlc]

  return tmp_df.reset_index()


def standardize(dataframe):
  scaler = StandardScaler()
  scaler.fit(dataframe)
  standardized_dataframe = scaler.transform(dataframe)

  return standardized_dataframe


def reduce_dimensionality(array):
  pca = PCA(n_components=0.95)
  pca.fit(array)
  dimesionality_reduced_array = pca.transform(array)

  return dimesionality_reduced_array

def convert_to_comma_separated_string(array):
  n_rows = array.shape[0]
  converted_array = []

  for line in range(0, n_rows):
    print(str(line) + "/" + str(n_rows))
    converted_list = [str(element) for element in array[line, :]]
    joined_string = ",".join(converted_list)
    converted_array.append(joined_string)

  return converted_array


def preprocess(dataframe, output_filepath):
  tmp_df = dataframe.copy()

  filtered_df = filter_by_dlc(tmp_df, 8)
  preprocessed_df = filtered_df.drop(columns=[consts.FLAG_COLUMN_NAME])

  standardized_dataframe = standardize(preprocessed_df)
  dimesionality_reduced_array = reduce_dimensionality(standardized_dataframe)

  # Standardize data again (according to the paper)
  dimesionality_reduced_array = standardize(dimesionality_reduced_array)
  dimesionality_reduced_array = np.round(dimesionality_reduced_array, 4)

  converted_array = convert_to_comma_separated_string(dimesionality_reduced_array)
  converted_df = pd.DataFrame(converted_array, columns = ['concatenated_features'])

  converted_df.to_csv(output_filepath, index=False)
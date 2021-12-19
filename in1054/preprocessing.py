import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import in1054.constants as consts

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
    converted_list = [str(element) for element in array[line, :]]
    joined_string = ",".join(converted_list)
    converted_array.append(joined_string)

  return converted_array


def preprocess(dataframe, output_filepath):
  preprocessed_df = dataframe.copy()
  preprocessed_df = preprocessed_df.drop(columns=[consts.FLAG_COLUMN_NAME])
  standardized_dataframe = standardize(dataframe)
  dimesionality_reduced_array = reduce_dimensionality(standardized_dataframe)
  # Standardize data again (according to the paper)
  dimesionality_reduced_array = standardize(dataframe)
  dimesionality_reduced_array = np.round(dimesionality_reduced_array, 4)
  converted_array = convert_to_comma_separated_string(dimesionality_reduced_array)

  with open(output_filepath, "w") as output:
    output.write(str(converted_array))
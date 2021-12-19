import pandas as pd
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


def preprocess(dataframe):
  preprocessed_df = dataframe.copy()
  preprocessed_df = preprocessed_df.drop(columns=[consts.FLAG_COLUMN_NAME])
  standardized_dataframe = standardize(dataframe)
  dimesionality_reduced_array = reduce_dimensionality(standardized_dataframe)

  return dimesionality_reduced_array
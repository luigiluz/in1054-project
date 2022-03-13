from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pandas as pd
import time
import numpy as np

import in1054.constants as consts

def conf_matrix(y_true, y_pred):

	return confusion_matrix(y_true, y_pred, normalize='all')

def classification_metrics_as_array(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1_score_metric = f1_score(y_true, y_pred)

	metrics_array = [[accuracy, precision, recall, f1_score_metric]]

	return metrics_array


def classification_metrics(y_true, y_pred):

	metrics_array = classification_metrics_as_array(y_true, y_pred)

	metrics_df = pd.DataFrame(metrics_array, columns=consts.METRICS_COLUMNS)

	return metrics_df


def bloom_filter_detection_time(input_data, bloomfilter):
	detection_time_list = []

	for i in range(50):
		start_time = time.time_ns()
		#bf_results = loaded_bloomfilter.multiple_query(test_data_merged)
		_ = bloomfilter.single_query(input_data[i])
		stop_time = time.time_ns()
		deltat = stop_time - start_time
		detection_time_list.append(deltat)

	mean_detection_time = np.mean(detection_time_list)

	return mean_detection_time

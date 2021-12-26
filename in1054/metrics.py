from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pandas as pd

import in1054.constants as consts

def conf_matrix(y_true, y_pred):

	return confusion_matrix(y_true, y_pred, normalize='all')

def classification_metrics(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1_score_metric = f1_score(y_true, y_pred)

	metrics_array = [[accuracy, precision, recall, f1_score_metric]]

	metrics_df = pd.DataFrame(metrics_array, columns=consts.METRICS_COLUMNS)

	return metrics_df
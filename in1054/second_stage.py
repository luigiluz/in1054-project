import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
import re
import string

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
	stripped_input = tf.strings.regex_replace(input_data,',',' ')

	return tf.strings.regex_replace(stripped_input,
									'[%s]' % re.escape(string.punctuation),
									'')


def convert_df_to_numpy_array(dataframe):

	return dataframe.to_numpy()


def create_text_vectorization_layer(text_data_array, output_path):
	# Create vectorizer
	vectorizer = TextVectorization(
									standardize=custom_standardization,
									output_mode="int",
									max_tokens=255,
									output_sequence_length=10
									)

	vectorizer.adapt(text_data_array)

	# Create model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
	model.add(vectorizer)

	model.save(output_path, save_format = "tf")


def load_text_vectorizer(vectorizer_path):

	loaded_model = tf.keras.models.load_model(vectorizer_path, compile=False)
	print("load_text_vectorizer: loaded_model")
	print(loaded_model.summary())
	loaded_vectorizer = loaded_model.layers[0]

	return loaded_vectorizer

# # TODO: Consider moving this to a layer of the neural network
# def _convert_text_to_tensor(text_data_array):
# 	vectorizer.adapt(text_data_array)
# 	binary_data = vectorizer(text_data_array)

# 	return binary_data


# def prepare_input_data(dataframe):
# 	tmp_df = dataframe.copy()

# 	df_as_np = _convert_df_to_numpy_array(tmp_df)
# 	#binary_data = _convert_text_to_tensor(df_as_np)

# 	return df_as_np


def create_second_stage_model(input_data_shape, model_hyperparameters):

	# Parameters parsing
	n_of_lstm_blocks = model_hyperparameters["n_of_lstm_blocks"]
	n_of_dense_neurons = model_hyperparameters["n_of_dense_neurons"]

	dropout_rate = model_hyperparameters["overfit_avoidance"]["dropout_rate"]
	regularizer_rate = model_hyperparameters["overfit_avoidance"]["regularizer_rate"]
	my_max_norm = model_hyperparameters["overfit_avoidance"]["max_norm"]

	my_learning_rate = model_hyperparameters["optimizer"]["learning_rate"]

	model = Sequential()

	# Add Input Layer
	model.add(keras.Input(shape=(input_data_shape)))
	model.add(keras.layers.Dropout(dropout_rate))

	# Add two initial hidden layers
	model.add(layers.Dense(n_of_dense_neurons,
							activation="relu",
							kernel_constraint=max_norm(my_max_norm),
							kernel_regularizer=regularizers.l2(regularizer_rate)))

	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(dropout_rate))

	model.add(layers.Dense(n_of_dense_neurons,
							activation="relu",
							kernel_constraint=max_norm(my_max_norm),
							kernel_regularizer=regularizers.l2(regularizer_rate)))

	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(dropout_rate))

	# Add two LSTM layers
	model.add(layers.Bidirectional(layers.LSTM(n_of_lstm_blocks,
									input_shape=(input_data_shape),
									return_sequences=True,
									kernel_regularizer=regularizers.l2(regularizer_rate))))

	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(dropout_rate))

	model.add(layers.Bidirectional(layers.LSTM(n_of_lstm_blocks)))

	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(dropout_rate))

	# Add output layer
	num_of_output_classes = 1
	model.add(layers.Dense(num_of_output_classes, activation="sigmoid"))

	# Compiles model
	opt = SGD(learning_rate=my_learning_rate) # ideal for now 0.000001
	model.compile(
				loss="binary_crossentropy",
				optimizer=opt,
				metrics=[
						tf.keras.metrics.BinaryAccuracy()
						#tf.keras.metrics.TruePositives()
						#tf.keras.metrics.Precision(),
						#tf.keras.metrics.Recall()
						]
						#["accuracy"]
				)

	return model


def train_second_stage_model(model, train_features, train_labels):

	model.fit(train_features, train_labels, batch_size=32, epochs=10)

	return model


def evaluate_second_stage_model(model, test_features, test_labels):

	model.evaluate(x=test_features, y=test_labels)


def second_stage_model_predict(model, test_features):

	predictions = model.predict(test_features)

	return predictions
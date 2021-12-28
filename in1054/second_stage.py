import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

def _convert_df_to_numpy_array(dataframe):

	return dataframe.to_numpy()


# TODO: Consider moving this to a layer of the neural network
def _convert_text_to_tensor(text_data_array):

	vectorizer = TextVectorization(output_mode="binary", ngrams=2)
	vectorizer.adapt(text_data_array)
	binary_data = vectorizer(text_data_array)

	return binary_data


def prepare_input_data(dataframe):
	tmp_df = dataframe.copy()

	df_as_np = _convert_df_to_numpy_array(tmp_df)
	binary_data = _convert_text_to_tensor(df_as_np)

	return binary_data


def create_second_stage_model():

	n_of_input_parameters = 51
	# Input layer
	inputs = keras.Input(shape=(n_of_input_parameters,))

	# Hidden layers
	x = layers.Dense(n_of_input_parameters, activation="relu")(inputs)
	# TODO: Add a second dense layer
	# TODO: Add a LSTM layer
	# TODO: Add a second LSTM layer (Output layer)

	# Output layer
	num_classes = 1
	outputs = layers.Dense(num_classes, activation="relu")(x)

	# Creates model
	model = keras.Model(inputs=inputs, outputs=outputs)

	# Compiles model
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model


def train_second_stage_model(model, train_features, train_labels):

	model.fit(train_features, train_labels, batch_size=32, epochs=10)

	return model


def evaluate_second_stage_model(model, test_features, test_labels):

	model.evaluate(x=test_features, y=test_labels)


def second_stage_model_predict(model, test_features):

	predictions = model.predict(test_features)

	return predictions
##### IMPORTS #####


# Third party imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####


def get_model_shapes(
        num_hidden_neurons,
        num_input_neurons,
        num_output_neurons
        ):
    """num_output_neurons
    Creates a list with the shapes of the weights and biases of the the model.

    Parameters:
        num_hidden_neurons (list): List of neurons of each hidden layer.
        num_input_neurons (int): Number of neurons of the input layer.
        num_output_neurons (int): Number of neurons of the output layer.

    Returns:
        shapes (list): Listo of shapes of the model.
    """
    shapes = []
    layer_sizes = [num_input_neurons] + num_hidden_neurons + [num_output_neurons]

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]

        shapes.append((in_dim, out_dim))  # pesos
        shapes.append((out_dim,))         # bias

    return shapes


def build_model(
        num_hidden_neurons,
        num_input_neurons,
        num_output_neurons
        ):
    """
    Function:
        Builds a neural network dense model given a the number of
        neruons of the dense layer.

    Parameters:
        num_hidden_neurons (list): Number of neurons of the dense layer.
        num_input_neurons (int): Number of neurons of the input layer.
        num_output_neurons (int): Number of neurons of the output layer.

    Returns:
        model: A neural network dense model.
    """
    model = Sequential()

    # First layer
    model.add(Dense(num_hidden_neurons[0], input_shape=(num_input_neurons,), activation='relu'))

    # Hidden layers
    for units in num_hidden_neurons[1:]:
        model.add(Dense(units, activation='rleu'))

    # Output layer
    model.add(Dense(num_output_neurons, activation='linear'))

    return model


def set_model_weights(model, weights, shapes):
    idx = 0
    new_weights = []
    for shape in shapes:
        size = np.prod(shape)
        array = np.array(weights[idx:idx+size]).reshape(shape)
        new_weights.append(array)
        idx += size

    model.set_weights(new_weights)


def test_set_model_weights():

    # Test the function
    model = build_model(
        NUM_HIDDEN_NEURONS,
        NUM_INPUT_NEURONS,
        NUM_OUTPUT_NEURONS
        )
    

    shapes = get_model_shapes(
        NUM_HIDDEN_NEURONS,
        NUM_INPUT_NEURONS,
        NUM_OUTPUT_NEURONS
        )
    
    print(shapes)
    print("\n")

    # Generate flat weight list with the correct total size: 2*5 + 5 + 5*1 + 1 = 21
    weights = np.arange(1, 22)  # [1, 2, ..., 21]


    # Set weights using your function
    set_model_weights(model, weights, shapes)

    # Get weights from the model to verify
    model_weights = model.get_weights()

    print(model_weights)

    # Flatten model weights to compare
    flattened = np.concatenate([w.flatten() for w in model_weights])

    # Assertion
    assert np.array_equal(flattened, weights), "Weights were not set correctly"
    print("Test passed: Weights correctly set.")

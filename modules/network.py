##### IMPORTS #####


# Third party imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####

def build_model(
        NUM_HIDDEN_NEURONS,
        NUM_INPUT_NEURONS,
        NUM_OUTPUT_NEURONS
        ):
    """
    Function:
        Builds a one layer neural network dense model given a the number of
        neruons of the dense layer.

    Parameters:
        NUM_HIDDEN_NEURONS (int): Number of neurons of the dense layer.
        NUM_INPUT_NEURONS (int): Number of neurons of the input layer.
        NUM_OUTPUT_NEURONS (int): Number of neurons of the output layer.

    Returns:
        model: A one layer neural network dense model.
    """
    model = Sequential([
        Dense(NUM_HIDDEN_NEURONS, input_shape=(NUM_INPUT_NEURONS,), activation='tanh'),
        Dense(NUM_OUTPUT_NEURONS, activation='linear')
    ])
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

    # Generate flat weight list with the correct total size: 2*5 + 5 + 5*1 + 1 = 21
    weights = np.arange(1, 22)  # [1, 2, ..., 21]

    # Set weights using your function
    set_model_weights(model, weights, shapes)

    # Get weights from the model to verify
    model_weights = model.get_weights()

    # Flatten model weights to compare
    flattened = np.concatenate([w.flatten() for w in model_weights])

    # Assertion
    assert np.array_equal(flattened, weights), "Weights were not set correctly"
    print("Test passed: Weights correctly set.")

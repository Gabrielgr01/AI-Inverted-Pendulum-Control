##### IMPORTS #####


# Third party imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd


# Local imports
from .config import *
from .utils import *


##### FUNCTIONS DEFINITION #####

def get_model_shapes(
        num_input_neurons,
        num_hidden_neurons,
        num_output_neurons
        ):
    """
    Function:
        Creates a list with the shapes of the weights and biases of the the model.

    Parameters:
        num_hidden_neurons (list): List of neurons of each hidden layer.
        num_input_neurons (int): Number of neurons of the input layer.
        num_output_neurons (int): Number of neurons of the output layer.

    Returns:
        shapes (list):  List of shapes of the model with the following format:
                        [(num of input neurons, num weigths for each neuron), 
                        (bias of input layer,), 
                        (num of neurons, num weigths for each neuron), 
                        (bias of hidden layer,), 
                        (num of output neurons, num weigths for each neuron), 
                        (bias of output layer,)]
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
        num_input_neurons,
        num_hidden_neurons,
        num_output_neurons,
        input_act_func,
        hidden_act_func,
        output_act_func
        ):
    """
    Function:
        Builds a dense neural network model given a number of neurons
        for the dense layer.

    Parameters:
        num_hidden_neurons (list): Number of neurons of the dense layer.
        num_input_neurons (int): Number of neurons of the input layer.
        num_output_neurons (int): Number of neurons of the output layer.
        input_act_funct (str): Activation function for the network's input layer neurons.
        hidden_act_funct (str): Activation function for the network's hidden layer neurons.
        output_act_funct (str): Activation function for the network's output layer neurons.

    Returns:
        model: A neural network dense model.
    """
    
    print("\n--> Creating the Neural Network ...\n")
    
    model = Sequential()

    # First layer
    model.add(Dense(num_hidden_neurons[0], 
                    input_shape=(num_input_neurons,), 
                    activation=input_act_func))

    # Hidden layers
    for units in num_hidden_neurons[1:]:
        model.add(Dense(units, activation=hidden_act_func))

    # Output layer
    model.add(Dense(num_output_neurons, activation=output_act_func))

    return model


def set_model_weights(model, weights):
    shapes = get_model_shapes(
        NUM_INPUT_NEURONS,
        NUM_HIDDEN_NEURONS,
        NUM_OUTPUT_NEURONS
    )    
    idx = 0
    new_weights = []
    for shape in shapes:
        size = np.prod(shape)
        array = np.array(weights[idx:idx+size]).reshape(shape)
        new_weights.append(array)
        idx += size

    model.set_weights(new_weights)

    return model


def model_predict(model, theta, vel):
    """
    Retorna el torque calculado por la red neuronal entrenada.
    
    Parámetros:
        model: Red neuronal entrenada (por ejemplo, un modelo de TensorFlow/Keras).
        theta (float): Ángulo actual del péndulo (en radianes).
        vel (float): Velocidad angular actual del péndulo (rad/s).
    
    Retorna:
        torque (float): Valor del torque de control calculado por la red.
    """
    
    max_df_vals, min_df_vals, df_differences = get_norm_config()

    # Normalizing inputs
    input_df = pd.DataFrame([[theta, vel]], columns=["theta", "vel"])
    input_norm_df = process_df("normalize", input_df, "", max_df_vals, min_df_vals, df_differences)
    theta_norm = input_norm_df["theta"].to_numpy()
    vel_norm = input_norm_df["vel"].to_numpy()
    model_input = np.column_stack((theta_norm, vel_norm))

    # Predicting the output torque
    torque = model.predict(model_input)
    
    # Denormalizing output
    output_df = pd.DataFrame([[torque]], columns=["torque"])
    output_norm_df = process_df("denormalize", output_df, "", max_df_vals, min_df_vals, df_differences)
    torque_denorm = output_norm_df["torque"].to_numpy()

    return float(torque_denorm)


def save_model(model, model_save_path):
    print ("\n--> Saving model ...\n")
    
    model.save(model_save_path)
    
    print("Model saved in:\n")
    print(model_save_path, "\n")


def load_model(MODEL_SAVE_PATH):
    print ("\n--> Loading model...\n")
    model = keras_load_model(MODEL_SAVE_PATH)
    print(f"Loaded model from:\n{MODEL_SAVE_PATH}\n")
    return model
    

##### TEST FUNCTIONS (For Developers) #####

def test_set_model_weights():

    # Test the function
    model = build_model(
        NUM_INPUT_NEURONS,
        NUM_HIDDEN_NEURONS,
        NUM_OUTPUT_NEURONS,
        INPUT_ACT_FUNCTION,
        HIDDEN_ACT_FUNCTION,
        OUTPUT_ACT_FUNCTION
        )
    

    shapes = get_model_shapes(
        NUM_INPUT_NEURONS,
        NUM_HIDDEN_NEURONS,
        NUM_OUTPUT_NEURONS
        )
    
    print("\nshapes: ", shapes)
    print("\n")

    # Generate flat weight list with the correct total size: 2*5 + 5 + 5*1 + 1 = 21
    weights = np.arange(1, 22)  # [1, 2, ..., 21]

    # Set weights using your function
    set_model_weights(model, weights, shapes)

    # Get weights from the model to verify
    model_weights = model.get_weights()

    print("Model weights: ")
    print(model_weights)
    print("\n")

    # Flatten model weights to compare
    flattened = np.concatenate([w.flatten() for w in model_weights])

    # Assertion
    assert np.array_equal(flattened, weights), "Weights were not set correctly"
    print("Test passed: Weights correctly set.")

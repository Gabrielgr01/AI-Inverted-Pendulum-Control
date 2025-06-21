##### IMPORTS #####


# Third party imports
import numpy as np


# Local imports
from .config import *
from .network import *


##### FUNCTIONS DEFINITION #####

def normalize_dataset(dataset):
    normalized = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            min_val = value.min()
            max_val = value.max()
            if max_val - min_val != 0:
                normalized[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized[key] = np.zeros_like(value)
        else:
            normalized[key] = value
    return normalized

def fitness_function(individual,
                     model,
                     dataset
                     ):
    
    # Creates the neural network
    set_model_weights(individual, model)

    dataset_torque = np.array(dataset["torque"])
    min_torque = dataset_torque.min()
    max_torque = dataset_torque.max()

    # Normalize the dataset
    norm_dataset = normalize_dataset(dataset)

    # Extract angle and velocity
    theta = np.array(norm_dataset["theta"])
    vel = np.array(norm_dataset["vel"])

    # Predict torque with the neural network
    model_input = np.column_stack((theta, vel))
    norm_network_torque = model.predict(model_input)

    # Denormalize the torque
    network_torque = norm_network_torque * (max_torque - min_torque) + min_torque

    # Calculate the fitness of the individual
    fitness = np.mean((np.array(dataset_torque) - np.array(network_torque)) ** 2)

    return fitness

def run_evolutionary_algorithm():
    
    print("")
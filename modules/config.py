##### IMPORTS #####

# Third party imports

# Built-in imports
import os

# Local imports


##### GLOBAL VARIABLES #####

RUN_AREA = os.getcwd()


PID_KP = 0.1
PID_KI = 0.2
PID_KD = 0.3


##### HYPERPARAMETERS #####

# Neural Network
NUM_INPUT_NEURONS = 2   # Angle and Velocity
NUM_OUTPUT_NEURONS = 1  # Torque
NUM_HIDDEN_LAYERS = 1   #
NUM_HIDDEN_NEURONS = 5  #
shapes = [
    (2, 5),  # Weights of from the input layer towards the hidden layer
    (5,),    # Bias of the hidden layer
    (5, 1),  # Weights of from the hidden layer towards the output layer
    (1,)     # Bias of the output layer
]

# Evolutionary Algorithm
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
DATASET_PATH = "results/dataset.csv"
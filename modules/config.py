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

# Evolutionary Algorithm
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
DATASET_PATH = "results/dataset.csv"
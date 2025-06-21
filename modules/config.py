##### IMPORTS #####

# Third party imports

# Built-in imports
import os

# Local imports


##### GLOBAL VARIABLES #####

RUN_AREA = os.getcwd() # Directory where the program is run
RESULTS_DIR_NAME = "results"
RESULTS_DIR_PATH = str(RUN_AREA) + f"\\{RESULTS_DIR_NAME}"
DATA_DIR_NAME = "data"
DATA_DIR_PATH = str(RUN_AREA) + f"\\{DATA_DIR_NAME}"

DATASET_CSV_NAME = "dataset.csv"
DATASET_CSV_PATH = str(DATA_DIR_PATH) + f"\\{DATASET_CSV_NAME}"

# PID Gains
PID_KP = 50 # Proportional Gain
PID_KI = 5  # Integral Gain
PID_KD = 10 # Derivative Gain

DYNAMIC_INPUT_PARAMS = [1.0,    # mass [kg]
                        1.0,    # length [m]
                        9.81,   # gravity [m/s^2]
                        0.1,    # friction coefficient
                        0.0     # external force (perturbation) [N]
                        ]
TARGET_ANGLE = 0 # [rad] Target angle for the pendulum control


##### HYPERPARAMETERS #####

# Neural Network
NUM_INPUT_NEURONS = 2   # 
NUM_OUTPUT_NEURONS = 1  # 
NUM_HIDDEN_LAYERS = 1   # 
NUM_HIDDEN_NEURONS = 5  #

shapes = [
    (2, 5),  # Weights of from the input layer towards the hidden layer
    (5,),    # Bias of the hidden layer
    (5, 1),  # Weights of from the hidden layer towards the output layer
    (1,)     # Bias of the output layer
]

# Evolutionary Algorithm
POPULATION_SIZE = 50    #
NUM_GENERATIONS = 100   #

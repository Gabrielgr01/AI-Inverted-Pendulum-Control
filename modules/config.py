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

# General
VERBOSE = False  
DEBUG = False    

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
NUM_INPUT_NEURONS = 3   # Number of neurons of the input layer
NUM_OUTPUT_NEURONS = 1  # Number of neurons of the output put layer

# List of neurons of each hidden layer. Example: [5, 5] represents two
# hidden layers of 5 neurons each.
NUM_HIDDEN_NEURONS = [5]

# Evolutionary Algorithm
POPULATION_SIZE = 100  # Population size
NUM_GENERATIONS = 30  # Number generations
PARENT_POPU_SIZE = 25  # Number of selected individuals
CHILD_POPU_SIZE = 25
MATE_CHANCE = 0.7
MUTATE_CHANCE = 0.3
MU = 0.0
GENE_RANGE = [-1, 1]
ALPHA = 0.5
SIGMA = (GENE_RANGE[1] - GENE_RANGE[0]) * 0.05

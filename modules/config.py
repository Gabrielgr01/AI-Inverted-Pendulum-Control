##### IMPORTS #####

# Third party imports

# Built-in imports
import os

# Local imports


##### GLOBAL VARIABLES #####

RUN_AREA = os.getcwd() # Directory where the program is run
RESULTS_DIR_NAME = "results"
RESULTS_DIR_PATH = os.path.join(RUN_AREA, RESULTS_DIR_NAME)
DATA_DIR_NAME = "data"
DATA_DIR_PATH = os.path.join(RUN_AREA, DATA_DIR_NAME)

DATASET_CSV_NAME = "dataset.csv"
DATASET_CSV_PATH = os.path.join(DATA_DIR_PATH, DATASET_CSV_NAME)

NETWORK_SIM_CSV_NAME = "network_sim.csv"
NETWORK_SIM_CSV_PATH = os.path.join(RESULTS_DIR_PATH, NETWORK_SIM_CSV_NAME)

MODEL_SAVE_NAME = "model.keras"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MODEL_SAVE_NAME)
MAX_CONFIG_SAVE_NAME = "max_config.csv"
MAX_CONFIG_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MAX_CONFIG_SAVE_NAME)
MIN_CONFIG_SAVE_NAME = "min_config.csv"
MIN_CONFIG_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MIN_CONFIG_SAVE_NAME)
MODEL_CONFIG_PATH_LIST = [MAX_CONFIG_SAVE_PATH, MIN_CONFIG_SAVE_PATH]

SIM_T_STOP = 10
DT = 0.01
SIM_T_SAMPLES = int((SIM_T_STOP / DT) + 1)

VERBOSE = False
DEBUG = False

PID_KP = 50         # Proportional Gain for the PID
PID_KI = 5          # Integral Gain for the PID
PID_KD = 10         # Derivative Gain for the PID
TARGET_ANGLE = 0    # [rad] Target angle for the pendulum control
DYNAMIC_INPUT_PARAMS = [1.0,    # mass [kg]
                        1.0,    # length [m]
                        9.81,   # gravity [m/s^2]
                        0.1     # friction coefficient
                        ]


##### HYPERPARAMETERS #####

# Neural Network
NUM_INPUT_NEURONS = 2   # Number of neurons of the input layer
NUM_OUTPUT_NEURONS = 1  # Number of neurons of the output put layer
NUM_HIDDEN_NEURONS = [5,] # List of neurons of each hidden layer. 
                          # Example: [5, 5] represents two hidden 
                          # layers of 5 neurons each.
INPUT_ACT_FUNCTION = "tanh"
HIDDEN_ACT_FUNCTION = "tanh"
OUTPUT_ACT_FUNCTION = "linear"

# Evolutionary Algorithm
POPULATION_SIZE = 100    #
NUM_GENERATIONS = 30   #
PARENT_POPU_SIZE = 60
CHILD_POPU_SIZE = 60
MATE_CHANCE = 0.7
MUTATE_CHANCE = 0.2
MU = 0.0
GENE_RANGE = [-1, 1]
ALPHA = 0.5
SIGMA = (GENE_RANGE[1] - GENE_RANGE[0]) * 0.01

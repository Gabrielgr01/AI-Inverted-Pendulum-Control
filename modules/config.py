##### IMPORTS #####

# Third party imports

# Built-in imports
import os

# Local imports


##### GLOBAL VARIABLES #####

RUN_AREA = os.getcwd() # Directory where the program is run
RESULTS_DIR_NAME = "results"
RESULTS_DIR_PATH = os.path.join(RUN_AREA, RESULTS_DIR_NAME)                 # Results directory path.
DATA_DIR_NAME = "data"
DATA_DIR_PATH = os.path.join(RUN_AREA, DATA_DIR_NAME)                       # Data directory path.

DATASET_CSV_NAME = "dataset.csv"
DATASET_CSV_PATH = os.path.join(DATA_DIR_PATH, DATASET_CSV_NAME)            # Dataset path.

NETWORK_SIM_CSV_NAME = "network_sim.csv"
NETWORK_SIM_CSV_PATH = os.path.join(RESULTS_DIR_PATH, NETWORK_SIM_CSV_NAME) # Neural network simulation data path.

MODEL_SAVE_NAME = "model.keras"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MODEL_SAVE_NAME)           # Neural network model path.
MAX_CONFIG_SAVE_NAME = "max_config.csv"
MAX_CONFIG_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MAX_CONFIG_SAVE_NAME) # Config path with max values for normalization.
MIN_CONFIG_SAVE_NAME = "min_config.csv"
MIN_CONFIG_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, MIN_CONFIG_SAVE_NAME) # Config path with min values for normalization.
MODEL_CONFIG_PATH_LIST = [MAX_CONFIG_SAVE_PATH, MIN_CONFIG_SAVE_PATH]       # List with max_config.csv and min_condig.csv paths.

SIM_T_STOP = 10 # Maximum simulation time
DT = 0.01       # Intervals length in the simulation
SIM_T_SAMPLES = int((SIM_T_STOP / DT) + 1) # Num of samples in the simulation.
PERTURBANCE_SIM_LIMIT = 5   # Max number of simulations with perturbance.
PERTURBANCE_SIM_COUNT = 0   # Count of simulations with perturbance.

VERBOSE = False     # To enable prints in the function.
DEBUG = False       # To print detailed messages useful for debugging.

PID_KP = 50         # Proportional Gain for the PID.
PID_KI = 5          # Integral Gain for the PID.
PID_KD = 10         # Derivative Gain for the PID.
TARGET_ANGLE = 0    # [rad] Target angle for the pendulum control.
DYNAMIC_INPUT_PARAMS = [1.0,    # mass [kg]
                        1.0,    # length [m]
                        9.81,   # gravity [m/s^2]
                        0.1     # friction coefficient
                        ]


##### HYPERPARAMETERS #####

# Neural Network
NUM_INPUT_NEURONS = 2       # Number of neurons of the input layer
NUM_OUTPUT_NEURONS = 1      # Number of neurons of the output put layer
NUM_HIDDEN_NEURONS = [5,]   # List of neurons of each hidden layer. 
                            # Example: [5, 5] represents two hidden 
                            # layers of 5 neurons each.
INPUT_ACT_FUNCTION = "tanh"     # Activation function for the neurons in the input layer
HIDDEN_ACT_FUNCTION = "tanh"    # Activation function for the neurons in the hidden layers
OUTPUT_ACT_FUNCTION = "linear"  # Activation function for the neurons in the output layer

# Evolutionary Algorithm
POPULATION_SIZE = 100   # Initial population size.
NUM_GENERATIONS = 30    # Number of generations to evolve.
PARENT_POPU_SIZE = 25   # Number of parents for each generation.
CHILD_POPU_SIZE = 25    # Number of childs for each generation.
MATE_CHANCE = 0.7       # Chance of mating to individuals.
MUTATE_CHANCE = 0.2     # Chance of mutating a new individual.
MU = 0.0                # Average of the normal distribution used in the mutation operator.
GENE_RANGE = [-1, 1]    # Valid range of values for the genes.
ALPHA = 0.5             # Recombination coeffient for mate operator.
SIGMA = (GENE_RANGE[1] - GENE_RANGE[0]) * 0.01 # Standard deviation for the mutation operator.

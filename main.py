##### IMPORTS #####

# Third party imports
import pandas as pd

# Built-in imports

# Local imports
from modules.config import *
import modules.simulation as sim
import modules.network as net
import modules.evolution as evol
import modules.utils as utils


##### FUNCTIONS DEFINITION #####

def train_new_model(dataset_dataframe):
    utils.create_directory(RESULTS_DIR_PATH, backup=False, overwrite=False)
    
    # Variables for data normalization
    max_df_vals = dataset_dataframe.max(axis=0) # Max of each dataset column
    min_df_vals = dataset_dataframe.min(axis=0) # Min of each dataset column
    norm_values = [max_df_vals, min_df_vals]
    
    # Creates the neural network model
    net_model = net.build_model(num_input_neurons=NUM_INPUT_NEURONS,
                                num_hidden_neurons=NUM_HIDDEN_NEURONS,
                                num_output_neurons=NUM_OUTPUT_NEURONS,
                                input_act_func=INPUT_ACT_FUNCTION,
                                hidden_act_func=HIDDEN_ACT_FUNCTION,
                                output_act_func=OUTPUT_ACT_FUNCTION)
    # Runs the evolutionary algorithm to get the network's weigths
    best_ind, best_ind_fitness = evol.run_evolutionary_algorithm(net_model, dataset_dataframe)

    if DEBUG == True:
        print("Best Individual: ", best_ind)
        print("Fitness: ", best_ind_fitness)

    net.set_model_weights(net_model, best_ind)
    
    return net_model, norm_values


##### MAIN EXECUTION #####

### TO TRAIN AND SAVE A NEW NEURAL NETWORK ###
#_, dataset_df = sim.generate_dataset(n_sims = 10)  # Generates new dataset of n_sims simulations
dataset_df = sim.load_dataset(DATASET_CSV_PATH)     # Loads dataset from .csv

new_model, norm_values = train_new_model(dataset_df)

net.save_model(new_model, MODEL_SAVE_PATH, MODEL_CONFIG_PATH_LIST, norm_values)
##############################################

# Loads the network model
net_model = net.load_model(MODEL_SAVE_PATH)

# Uses the network model to predict
theta = 0.2
vel = -0.5
torque = net.model_predict(net_model, theta, vel, MODEL_CONFIG_PATH_LIST)
print ("Torque: ", torque)

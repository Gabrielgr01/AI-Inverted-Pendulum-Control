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


##### GLOBAL VARIABLES #####


##### FUNCTIONS DEFINITION #####

def train_new_model(dataset_dataframe):
    utils.create_directory(RESULTS_DIR_PATH, backup=False, overwrite=False)
    
    # Creates the neural network model
    net_model = net.build_model(num_input_neurons=NUM_INPUT_NEURONS,
                                num_hidden_neurons=NUM_HIDDEN_NEURONS,
                                num_output_neurons=NUM_OUTPUT_NEURONS,
                                input_act_func=INPUT_ACT_FUNCTION,
                                hidden_act_func=HIDDEN_ACT_FUNCTION,
                                output_act_func=OUTPUT_ACT_FUNCTION)
    # Runs the evolutionary algorithm to get the network's weigths
    best_ind, best_ind_fitness = evol.run_evolutionary_algorithm(net_model, dataset_dataframe)

    print("Best Individual: ", best_ind)
    print("Fitness: ", float(best_ind_fitness))

    net.set_model_weights(net_model, best_ind)
    
    return net_model


##### MAIN EXECUTION #####

### TO TRAIN AND SAVE A NEW NEURAL NETWORK ###
#_, dataset_df = sim.generate_dataset(n_sims = 20, data_path=DATA_DIR_PATH, option="PID")  # Generates new dataset of n_sims simulations
#dataset_df = sim.load_dataset(DATASET_CSV_PATH)     # Loads dataset from .csv

#new_model = train_new_model(dataset_df)

#net.save_model(new_model, MODEL_SAVE_PATH)
##############################################

# Loads the network model
net_model = net.load_model(MODEL_SAVE_PATH)

# Uses the network model to predict
########################
#theta = 0.2
#vel = -0.5
#torque = net.model_predict(net_model, theta, vel)
#print ("Torque: ", torque)
########################

_, network_sim_df = sim.generate_dataset(n_sims = 10, data_path=RESULTS_DIR_PATH, option="NETWORK", model=net_model)

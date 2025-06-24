##### IMPORTS #####

# Third party imports

# Built-in imports

# Local imports
from modules.config import *
import modules.utils as util
import modules.simulation as sim
import modules.network as net
import modules.evolution as evol


##### GLOBAL VARIABLES #####


##### FUNCTIONS DEFINITION #####

def train_new_model(dataset_dataframe):
    util.create_directory(RESULTS_DIR_PATH, backup=False, overwrite=False)
    
    # Creates the neural network model
    net_model = net.build_model(num_input_neurons=NUM_INPUT_NEURONS,
                                num_hidden_neurons=NUM_HIDDEN_NEURONS,
                                num_output_neurons=NUM_OUTPUT_NEURONS,
                                input_act_func=INPUT_ACT_FUNCTION,
                                hidden_act_func=HIDDEN_ACT_FUNCTION,
                                output_act_func=OUTPUT_ACT_FUNCTION)
    # Runs the evolutionary algorithm to get the network's weigths
    best_ind, best_ind_fitness = evol.run_evolutionary_algorithm(net_model, dataset_dataframe)

    net.set_model_weights(net_model, best_ind)
    
    print("Best Individual: ", best_ind)
    print("Fitness: ", best_ind_fitness[0])
    
    return net_model


##### MAIN EXECUTION #####

## Generates new dataset of n_sims simulations
_, dataset_df = sim.generate_dataset(n_sims = 10, data_path=DATA_DIR_PATH, option="PID")

## Loads a dataset from a .csv file
#dataset_df = sim.load_dataset(DATASET_CSV_PATH)

## Trains the new neural network model
new_model = train_new_model(dataset_df)

## Saves the neural network model
net.save_model(new_model, MODEL_SAVE_PATH)

## Loads the neural network model
net_model = net.load_model(MODEL_SAVE_PATH)

## Simulates the neural network model
_, network_sim_df = sim.generate_dataset(n_sims = 10, data_path=RESULTS_DIR_PATH, option="NETWORK", model=net_model)

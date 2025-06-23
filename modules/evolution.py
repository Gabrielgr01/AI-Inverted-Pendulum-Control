##### IMPORTS #####


# Third party imports
import numpy as np
from deap import creator, base, tools, algorithms
import random
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

# Local imports
from .config import *
from .network import *
from .simulation import *
from .visualization import *


##### FUNCTIONS DEFINITION #####

toolbox = base.Toolbox()

### Not used
def count_weights(
        num_input_neurons,
        num_hidden_neurons,
        num_output_neurons
        ):
    """
    Function:
        Finds the total number of chromosomes.

    Parameters:
        NUM_HIDDEN_NEURONS (list): Number of neurons of the dense layer.
        NUM_INPUT_NEURONS (int): Number of neurons of the input layer.
        NUM_OUTPUT_NEURONS (int): Number of neurons of the output layer.

    Returns:
        total_weights (int): Total number of weights, which represents the
                             total number of chromosomes.
    """
    layer_sizes = [num_input_neurons] + num_hidden_neurons + [num_output_neurons]
    total_weights = 0

    for i in range(len(layer_sizes) - 1):
        w = layer_sizes[i] * layer_sizes[i + 1]  # pesos entre capas
        b = layer_sizes[i + 1]                   # bias en capa siguiente
        total_weights += w + b

    return total_weights
###

def get_model_num_weights(net_model):
    weights = net_model.get_weights()
    total_weights_num = 0
    for weight in weights:
        total_weights_num += weight.size
        #print("size: ", weight.size)
    return total_weights_num


def normalize(df):
    """
    Function:
        Normalize numeric columns in a pandas DataFrame to range [0, 1].

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        normalized_df (pd.DataFrame): A new DataFrame with normalized numeric columns.
    """
    normalized_df = df.copy()

    for column in df.select_dtypes(include=[np.number]).columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val - min_val != 0:
            normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            normalized_df[column] = 0.0  # all values are the same

    return normalized_df


def evaluation_function(
        individual,
        model,
        dataset_df
        ):

    # Sets the model with the weights defined by the individual
    set_model_weights(model, individual)

    torque_array = dataset_df["torque"].to_numpy()
    min_torque = torque_array.min()
    max_torque = torque_array.max()

    # Normalize the dataset
    norm_dataset = normalize(dataset_df)

    # Extract angle and velocity
    theta_norm = norm_dataset["theta"].to_numpy()
    vel_norm = norm_dataset["vel"].to_numpy()

    # Predict torque with the neural network
    model_input = np.column_stack((theta_norm, vel_norm))
    norm_network_torque = model.predict(model_input)

    # Denormalize the torque
    network_torque = norm_network_torque * (max_torque - min_torque) + min_torque

    # Calculate the fitness of the individual
    fitness = np.mean((np.array(torque_array) - np.array(network_torque)) ** 2)

    return (fitness,)


def run_evolutionary_algorithm(net_model, dataset_df):
    """
    Function:
        Runs a multi-objective evolutionary algorithm using the DEAP library.

        This function sets up and executes an evolutionary process to optimize two 
        conflicting objectives (displacement and acceleration). The function registers 
        genetic operators, initializes the population, executes the evolution, and 
        visualizes the results.

    Parameters:
        None

    Returns:
        best_ind (list): List with the trained weights of the neural network.
        best_ind_fitness (float) : Fitness value of the 'best_ind'.
    """
    
    print("\n--> Running the Evolutionary Algorithm ...\n")

    # Negative weights for minimization
    pesos_fitness = (-1.0,)

    n_genes = get_model_num_weights(net_model)

    # Fitness function definition
    creator.create("fitness_function", base.Fitness, weights=pesos_fitness)
    # Individual definition
    creator.create("individual", list, fitness=creator.fitness_function, typecode="f")

    # Alleles
    toolbox.register("gene", random.uniform, GENE_RANGE[0], GENE_RANGE[1])

    # Individual generator
    toolbox.register(
        "individual_generation",
        tools.initRepeat,
        creator.individual,
        toolbox.gene,
        n_genes,
    )

    # Population generator
    toolbox.register(
        "population", tools.initRepeat, list, toolbox.individual_generation
    )

    toolbox.register("evaluate", partial(evaluation_function, model=net_model, dataset_df=dataset_df))
    # Evolution operators
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxBlend, alpha=ALPHA)
    toolbox.register(
        "mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=0.2
    )

    if DEBUG == True:
        # Test for the population and individuals generation
        population_test = toolbox.population(n=POPULATION_SIZE)
        individual_test = toolbox.individual_generation()
        print("Individuo: ", individual_test)
        print("Ejemplo de poblacion: ", population_test)

    # Statistics on the general fitness of the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)  # Generation 'Average'
    stats.register("std", np.std)  # Individuals 'Standard Deviation'
    stats.register("min", np.min)  # 'Min Fitness' of the generation
    stats.register("max", np.max)  # 'Max Fitness' of the generation

    hof = tools.HallOfFame(1)  # Hall of Fame
    popu = toolbox.population(n=POPULATION_SIZE)  # Defines the initial population
    
    # Runs the Evolutionary Algorithm
    popu, logbook = algorithms.eaMuPlusLambda(
        population=popu,
        toolbox=toolbox,
        mu=PARENT_POPU_SIZE,
        lambda_=CHILD_POPU_SIZE,
        cxpb=MATE_CHANCE,
        mutpb=MUTATE_CHANCE,
        ngen=NUM_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=VERBOSE,
    )

    best_ind = hof[0]
    best_ind_fitness = best_ind.fitness.values
    log_df = pd.DataFrame(logbook)
    
    # View last generation summary
    print()
    print()
    print(log_df.tail(1))
    print("")

    # Plot fitness across generations
    y_fitness_dict = {
        "Min Fitness" :  log_df["min"],
        "Avg Fitness" : log_df["avg"]
    }
    create_multi_y_graph(
        x_values = log_df["gen"],
        x_title = "Generation",
        y_values_dict = y_fitness_dict,
        plot_type = "plot",
        show_plot = True,
        graph_title = "Fitness over Generations",
        image_name = "evolution_fitness",
        image_path = RUN_AREA
    )

    return best_ind, best_ind_fitness

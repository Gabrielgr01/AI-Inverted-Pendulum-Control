##### IMPORTS #####


# Third party imports
import numpy as np
from deap import creator, base, tools, algorithms
import random
import pandas as pd
from functools import partial

# Local imports
from .config import *
from .utils import *
from .network import *
from .simulation import *


##### FUNCTIONS DEFINITION #####

toolbox = base.Toolbox()


def get_model_num_weights(net_model):
    """
    Function:
        Sums the total number of weights and biases of the neural network.

    Parameters:
        net_model (model.keras): Neural network model.

    Returns:
        total_weights_num (int): Total number of weights and biases.
    """
    
    weights = net_model.get_weights()
    total_weights_num = 0
    for weight in weights:
        total_weights_num += weight.size
    return total_weights_num


def evaluation_function(
        individual,
        model,
        dataset_df
        ):

    """
    Function:
        Evaluation function used to set the fitness of an individual.

    Parameters:
        individual (list): Defined with the neural network possible weights.
        model (model.keras): Neural network model.
        dataset_df (pd.DataFrame): The dataset to which compare the model prediction.

    Returns:
        fitness (array): Fitness value of the individual.
    """

    # Sets the model with the weights defined by the individual
    set_model_weights(model, individual)

    torque_array = dataset_df["torque"].to_numpy()
    
    max_df_vals, min_df_vals = get_norm_config()
    
    # Extract angle and velocity
    theta_norm = dataset_df["theta"].to_numpy()
    vel_norm = dataset_df["vel"].to_numpy()

    # Predict torque with the neural network
    model_input = np.column_stack((theta_norm, vel_norm))
    norm_network_torque = model.predict(model_input)

    # Denormalize the torque
    norm_net_torque_df = pd.DataFrame(norm_network_torque, columns=["torque"])
    norm_dataset = process_df("denormalize", norm_net_torque_df, "", max_df_vals, min_df_vals)
    network_torque = norm_dataset["torque"]
    
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
    toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=0.2)

    if DEBUG == True:
        # Test for the population and individuals generation
        population_test = toolbox.population(n=POPULATION_SIZE)
        individual_test = toolbox.individual_generation()
        print("Individuo: ", individual_test)
        print("Ejemplo de poblacion: ", population_test)

    # Statistics on the general fitness of the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean) # Generation 'Average'
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
    print("Last generation: ")
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
        image_path = RESULTS_DIR_PATH
    )

    return best_ind, best_ind_fitness

##### IMPORTS #####

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt

# Built-in imports
import shutil
import os

# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####

def create_directory(full_path, backup=False, overwrite=True):
    """
    Function: 
        Manages directories creation.

    Parameters:
        full_path (str): Full path of the directory to create.
        backup (bool): If the directory was to be overwritten, creates a 
            backup with an "_old" suffix.
        overwrite (bool): Wheter to overwrite the directory or not if it
            exists.

    Returns:
        None
    """

    if backup == True:
        if os.path.exists(full_path):
            old_path = full_path + "_old"
            if os.path.exists(old_path):
                shutil.rmtree(old_path)
            os.replace(full_path, old_path)
        else:
            os.makedirs(full_path)
    else:
        if os.path.exists(full_path):
            if overwrite == True:
                shutil.rmtree(full_path)
                os.makedirs(full_path)
        else:
            os.makedirs(full_path)


def print_files_tree():
    print("""
Program File Tree:
C:.
|   main.py
|   
|
|-- data
|       dataset.csv
|       inv_pen_sim_1.png
|       inv_pen_sim_2.png
|       ...
|
|-- modules
|       config.py
|       evolution.py
|       network.py
|       simulation.py
|       utils.py
|       __init__.py
|
|-- results
        evolution_fitness.png
        inv_pen_sim_1.png
        inv_pen_sim_2.png
        ...
        max_config.csv
        min_config.csv
        model.keras
        network_sim.csv
    """)


def print_files_tree_short():
    """
    Function: 
        Prints the hierarchical tree structure of the main 
        program files and modules. Includes only code files.
        Used to explain the code in the written report.
    
    Parameters:
        None

    Returns:
        None
    """
    
    print("""
Program File Tree:
C:.
│   main.py
│
└───modules
        config.py
        evolution.py
        network.py
        simulation.py
        utils.py
        visualization.py
        __init__.py
          """)


def process_df(operation, df, title, max_vals, min_vals):
    """
    Function:
        Normalizes or denormalizes values of a pd.DataFrame.

    Parámetros:
        operation (str): To choose between 'normalize' o 'denormalize'.
        df (Pandas.DataFrame): DataFrame to process.
        title (str): Title to print in debug mode.
        max_vals (array): max values of the DataFrame for the normalization.
        min_vals (array): min values of the DataFrame for the normalization.

    Retorna:
        norm_df (Pandas.DataFrame): (De)normalized DataFrame.
    """
    
    difference = max_vals - min_vals
    
    if operation == "normalize":
        # Normalización tomando en cuenta caso donde toda la columna tiene
        # elementos iguales
        norm_df = df.apply(
            lambda col: (
                (col - min_vals[col.name]) / difference[col.name]
                if difference[col.name] != 0
                else pd.Series([0.0] * len(col), index=col.index)
            )
        )
    elif operation == "denormalize":
        # Des-normalización tomando en cuenta caso donde toda la columna tiene
        # elementos iguales
        norm_df = df.apply(
            lambda col: (
                col * difference[col.name] + min_vals[col.name]
                if difference[col.name] != 0
                else pd.Series([min_vals[col.name]] * len(col), index=col.index)
            )
        )
    else:
        raise ValueError(
            "-I-: 'operation' parameter should be one of the following: 'normalize' or 'denormalize'."
        )

    norm_df = norm_df.astype(float)

    if DEBUG == 1:
        title = operation.capitalize() + "d " + title + ":"
        print(title)
        print(norm_df)

    return norm_df


def save_norm_config(max_vals, min_vals):
    """
    Function: 
        Saves a configuration file with the max and min values
        used in the dataset normalization for future reference.
        
        Files are saved in the MODEL_CONFIG_PATH_LIST paths 
        ('results' directory).

    Parameters:
        max_vals (array): max values of the DataFrame for the normalization.
        min_vals (array): min values of the DataFrame for the normalization.

    Returns:
        None
    """
    
    print("\n--> Saving normalization data ...")
    
    max_config_path = MODEL_CONFIG_PATH_LIST[0]
    min_config_path = MODEL_CONFIG_PATH_LIST[1]
    
    max_vals = max_vals.to_frame().T
    min_vals = min_vals.to_frame().T
    max_vals.to_csv(max_config_path, index=False)
    min_vals.to_csv(min_config_path, index=False)
    
    print("Normalization configs created: ")
    print(MODEL_CONFIG_PATH_LIST[0])
    print(MODEL_CONFIG_PATH_LIST[1], "\n")
    

def get_norm_config():
    """
    Function: 
        Returns the max and min values read from the normalization 
        config files.

    Parameters:
        None

    Returns:
        max_df_vals (array): max values of the DataFrame for the normalization.
        min_df_vals (array): min values of the DataFrame for the normalization.
    """
    
    max_df_recuperado = pd.read_csv(MODEL_CONFIG_PATH_LIST[0])
    min_df_recuperado = pd.read_csv(MODEL_CONFIG_PATH_LIST[1])
    max_df_vals = max_df_recuperado.iloc[0]
    min_df_vals = min_df_recuperado.iloc[0]
    
    return max_df_vals, min_df_vals


def create_simple_graph(
    x_values,
    x_title,
    y_values,
    y_title,
    annotate_values,
    plot_type,
    show_plot,
    graph_title,
    image_name,
    image_path,
):
    """
    Function: 
        Creates and saves a 2D graph (scatter or line).

    Parameters:
        x_values (list): Data points for the x-axis.
        x_title (str): Label for the x-axis.
        y_values (list): Data points for the y-axis.
        y_title (str): Label for the y-axis.
        annotate_values (list): Optional. List of two lists
                                [k_values, b_values] for annotating each
                                (k, b) point.
        plot_type (str): Type of plot to create. Valid options: 'scatter',
                            others default to line plot.
        show_plot (bool): Shows the plot while running if True.
        graph_title (str): Title of the graph.
        image_name (str): Name of the image to generate.
        image_path (str): Path to save the image.

    Returns:
        None
    """

    plt.figure()
    match plot_type:
        case "scatter":
            plt.scatter(x_values, y_values)
        case "plot":
            plt.plot(x_values, y_values)
        case _:
            print(
                "-W-: For 'create_multi_y_graph' attribute 'plot_type' use: scatter or plot"
            )

    # Annotate each point with k and b values
    if len(annotate_values) == 2:
        k_values = annotate_values[0]
        b_values = annotate_values[1]
        for i in range(len(x_values)):
            plt.annotate(
                f"k={k_values[i]:.1f}\nb={b_values[i]:.1f}",
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
            )

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    if show_plot == True:
        plt.show()
    plt.close()


def create_multi_y_graph(
    x_values,
    x_title,
    y_values_dict,
    plot_type,
    show_plot,
    graph_title,
    image_name,
    image_path,
):
    """
    Function: 
        Creates and saves a 2D graph with multiple y-axis functions
        (scatter or line).

    Parameters:
        x_values (list or array): Data points for the x-axis.
        x_title (str): Label for the x-axis.
        y_values_dict (dict): Dictionary where keys are labels and values are
                              lists of y data.
        plot_type (str): Type of plot to create. Valid options: 'scatter',
                         others default to line plot.
        show_plot (bool): Shows the plot while running if True.
        graph_title (str): Title of the graph.
        image_name (str): Name of the image to generate.
        image_path (str): Path to save the image.

    Returns:
        None
    """

    plt.figure()
    for y_title, y_values in y_values_dict.items():
        match plot_type:
            case "scatter":
                plt.scatter(x_values, y_values, label=y_title)
            case "plot":
                plt.plot(x_values, y_values, label=y_title)
            case _:
                print(
                    "-W-: For 'create_multi_y_graph' attribute 'plot_type' use: scatter or plot"
                )
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    if show_plot == True:
        plt.show()
    plt.close()

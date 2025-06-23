##### IMPORTS #####

# Third party imports
import pandas as pd

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
        backup (bool):  If the directory was to be overwritten, creates a 
                        backup with an "_old" suffix.

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

def process_df(operation, df, title, max_val, min_val, difference):
    """
    Función para normalizar o desnormalizar un DataFrame.

    Parámetros:
        operation (str): 'normalize' o 'denormalize' para elegir la operación.
        df (Pandas.DataFrame): DataFrame a procesar.
        title (str): título para los datos a imprimir en modo debug.

    Retorna:
        norm_df (Pandas.DataFrame): DataFrame normalizado o desnormalizado.
    """
    
    if operation == "normalize":
        # Normalización tomando en cuenta caso donde toda la columna tiene
        # elementos iguales
        norm_df = df.apply(
            lambda col: (
                (col - min_val[col.name]) / difference[col.name]
                if difference[col.name] != 0
                else pd.Series([0.0] * len(col), index=col.index)
            )
        )
    elif operation == "denormalize":
        # Des-normalización tomando en cuenta caso donde toda la columna tiene
        # elementos iguales
        norm_df = df.apply(
            lambda col: (
                col * difference[col.name] + min_val[col.name]
                if difference[col.name] != 0
                else pd.Series([min_val[col.name]] * len(col), index=col.index)
            )
        )
    else:
        raise ValueError(
            "El parámetro 'operation' debe ser 'normalize' o 'denormalize'."
        )

    norm_df = norm_df.astype(float)

    if DEBUG == 1:
        title = operation.capitalize() + "d " + title + ":"
        print(title)
        print(norm_df)

    return norm_df

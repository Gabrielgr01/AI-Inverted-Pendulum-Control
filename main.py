import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Importar funciones locales
from modules.config import *
import modules.simulation as sim



##### MAIN EXECUTION #####

# Generates dataset of 10 simulations
sim.generate_dataset(
    n_sims = 10
)

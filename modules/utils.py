##### IMPORTS #####

# Third party imports

# Built-in imports
import shutil

# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####

def create_directory(full_path, backup=False):
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
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
    
    os.makedirs(full_path)


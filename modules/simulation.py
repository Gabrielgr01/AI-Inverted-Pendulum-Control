##### IMPORTS #####

# Third party imports
import numpy as np
from scipy.integrate import odeint
import pandas as pd

# Built-in imports
import random

# Local imports
from .config import *
from .utils import *
from .network import *


##### FUNCTIONS DEFINITION #####

def get_inv_pendulum_acceleration(input_params, theta, vel, torque_control, f_ext):
    """
    Function: 
        Defines the differential equation for a inverted pendulum system.
        Solves for acceleration.

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        theta (float): Angle value.
        vel (float): Velocity value.
        torque_control (float): Control torque applied to the system.
        f_ext (float): External force (perturbance).

    Returns:
        acc (float): Acceleration of the system.
    """
    
    m = input_params[0]
    l = input_params[1]
    g = input_params[2]
    B = input_params[3]
    
    if torque_control == 0:
        acc = + f_ext*(1/m*l) - np.sin(theta)*(g/l) + vel*(B/m*l**2)
    else:
        acc = torque_control*(1/m*l**2) - f_ext*(1/m*l) - np.sin(theta)*(g/l) - vel*(B/m*l**2)
    return acc


def get_inv_pendulum_torque(input_params, theta, vel, acc, f_ext):
    """
    Function: 
        Defines the differential equation for a inverted pendulum system.
        Solves for torque.

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        theta (float): Angle value.
        vel (float): Velocity value.
        acc (float): Acceleration value.
        f_ext (float): External force (perturbance).

    Returns:
        torque (float): Applied control torque of the system.
    """
    
    m = input_params[0]
    l = input_params[1]
    g = input_params[2]
    B = input_params[3]
    f_ext = input_params[4]
    
    torque = m*l**2*acc + m*g*l*np.sin(theta) + B*vel + f_ext*l
    return torque


def get_inv_pendulum_model(S, t, input_params, torque_control, f_ext):
    """
    Function: 
        Defines the differential state equations for a inverted pendulum system.

    Parameters:
        S (list): State vector / Initial conditions [angle 'theta', velocity 'vel'].
        t (float): Time variable (not used explicitly but required by odeint).
        input_params (list): Physical constants for the differential ecuation.
        torque_control (float): System control torque.

    Returns:
        List: Derivatives [dtheta/dt, dv/dt].
    """
       
    theta, vel = S
    return [vel, get_inv_pendulum_acceleration(input_params, theta, vel, torque_control, f_ext)]


def solve_inv_pendulum_model(input_params, torque_control, initial_state, t_start, t_stop, t_samples, f_ext):
    """
    Function: 
        Solves the inverted pendulum system using the given parameters.

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        torque_control (float): System control torque.
        t_start (float): Starting time for the simulation.
        t_stop (float): Maximum time for simulation.
        t_samples (int): Number of time samples.

    Returns:
        List: time (t), angle (theta), velocity (v), and acceleration (a).
    """
    
    S_0 = initial_state
    t = np.linspace(t_start, t_stop, t_samples)
    
    solution = odeint(get_inv_pendulum_model, y0=S_0, t=t, args=(input_params, torque_control, f_ext))
    theta_sol = solution.T[0]
    vel_sol = solution.T[1]
    acc_sol = get_inv_pendulum_acceleration(input_params, theta_sol, vel_sol, torque_control, f_ext)

    return t, theta_sol, vel_sol, acc_sol


def impulse(dt, current_t, t_impulse, magnitude):
    """
    Function:
        Impulse function.

    Parameters:
        dt (float): Time interval for each step of the simulation.
        current_t (float): Current simulation time.
        t_impulse (float): Time when the impulse should appear.
        magnitude (float): Magnitude of the impulse.

    Returns:
        _ (float): Impulse magnitude
    """
    
    if abs(current_t - t_impulse) <= dt:
        return magnitude
    else:
        return 0


def pid_control(kp, ki, kd, target, inputs, prev_integral_error, prev_error, dt):
    """
    Function:
        PID controller.

    Parameters:
        kp, ki, kd (float): PID gains.
        target (float): Desired/Reference value.
        inputs (list): current system value (ex. angle, velocity)
        prev_integral_error (float): cummulative sum of the error (integral)
        prev_error (float): previous step error (derivative)
        dt (float): time step

    Returns:
        output: controlled output
        integral_error: updated integral error
        error: updated error (prev_error for the next iteration)
    """
    error = target - inputs
    integral_error = prev_integral_error + error * dt
    derivative_error = (error - prev_error) / dt

    output = (kp * error) + (ki * integral_error) + (kd * derivative_error) # Output is torque

    return output, integral_error, error


def get_pid_gains(input_params, init_conditions, target = 0, t_max = 5, t_samples = 300):
    """
    Function:
        Plots a series of graphs with the PID behavior for several gain combinations.
        This function is only run once for the PID calibration.

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        init_conditions (list): Initial 'theta' and 'vel' values.
        target (float): Desired/Reference value.
        t_max (float): Maximum time for simulation.
        t_samples (int): Number of time samples.

    Returns:
        None
    """
    
    input_params = DYNAMIC_INPUT_PARAMS
    init_conditions = [0.2, 0.0]  # angle, velocity

    # Gain combinations to test (Kp, Ki, Kd)
    pid_tests = [
        (10, 0, 0),   # Pure proporcional gain
        (20, 0, 5),   # Proporcional + derivative
        (30, 2, 8),   # Agressive PID
        (50, 5, 10),  # Strong PID
        (15, 1, 2),   # Soft PID
    ]

    for i, pid_gains in enumerate(pid_tests):
        data = get_simulated_data(input_params, target, init_conditions, pid_gains, t_max, t_samples)

        t = data['t']
        theta = data['theta']
        vel = data['vel']
        torque = data['torque']

        model_solutions = {
            'theta': theta,
            'vel': vel,
            'torque': torque
        }

        create_multi_y_graph(
            x_values = t,
            x_title = "Tiempo [s]",
            y_values_dict = model_solutions,
            plot_type = "scatter",
            show_plot = True,
            graph_title = f"PID Test {i+1} - Kp={pid_gains[0]}, Ki={pid_gains[1]}, Kd={pid_gains[2]}",
            image_name = f"pid_test_{i+1}",
            image_path = RUN_AREA,
        )
        
        create_simple_graph(
            x_values = t,
            x_title = "Tiempo",
            y_values = model_solutions['theta'],
            y_title = "Theta",
            annotate_values = [],
            plot_type = "scatter",
            show_plot = True,
            graph_title = f"PID Test Theta {i+1} - Kp={pid_gains[0]}, Ki={pid_gains[1]}, Kd={pid_gains[2]}",
            image_name = f"pid_test_theta_{i+1}",
            image_path = RUN_AREA,
        )


def get_simulated_data(input_params, target, init_conditions, pid_gains, t_max, t_samples, perturbance):
    """
    Function:
        Simulates the inverted pendulum system control (PID).

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        target (float): Desired/Reference value.
        init_conditions (list): Initial 'theta' and 'vel' values.
        pid_gains (list): Gain constants for the PID controller.
        t_max (float): Maximum time for simulation.
        t_samples (int): Number of time samples.
        perturbance (bool): Indicates the presence of a perturbance in the simulation.

    Returns:
        dataset (dict):
            't': time (array)
            'theta': angle (array)
            'vel': angular velocity (array)
            'acc': angular acceleration (array)
            'torque': control torque (array)
    """
    global PERTURBANCE_SIM_COUNT

    dt = t_max / (t_samples - 1)
    t = np.linspace(0, t_max, t_samples)

    # Initial Conditions
    theta, vel = init_conditions

    # PID parameters
    integral_error = 0.0
    prev_error = 0.0 
    kp = pid_gains[0]
    ki = pid_gains[1]
    kd = pid_gains[2]

    # Lists to save the results
    theta_list = []
    vel_list = []
    acc_list = []
    torque_list = []
    
    if perturbance == True and PERTURBANCE_SIM_COUNT < PERTURBANCE_SIM_LIMIT:
        t_impulse = random.uniform(0.0, t_max/2)
        PERTURBANCE_SIM_COUNT += 1
    else:
        t_impulse = t_max + 1

    for i in range(t_samples):
        t_seconds = (i/t_samples) * t_max
        magnitude = random.uniform(-100, 100)
        f_ext = impulse(dt, t_seconds, t_impulse, magnitude)
        
        torque, integral_error, prev_error = pid_control(
            kp, ki, kd, target, theta, integral_error, prev_error, dt
        )
        acc = get_inv_pendulum_acceleration(input_params, theta, vel, torque, 0)

        theta_list.append(theta)
        vel_list.append(vel)
        acc_list.append(acc)
        torque_list.append(torque)
        
        # Solve system with small time step
        S_0 = [theta, vel]
        _, theta_sol, vel_sol, acc_sol = solve_inv_pendulum_model(input_params, torque, S_0, 0, dt, 2, f_ext)
        
        theta = theta_sol[-1]   # Gets the last calculated theta
        vel = vel_sol[-1]       # Gets the last calculated velocity
        acc = acc_sol[-1]       # Gets the last calculated acceleration
        
    sim_data = {
        't': t,
        'theta': np.array(theta_list),
        'vel': np.array(vel_list),
        'acc': np.array(acc_list),
        'torque': np.array(torque_list)
    }

    return sim_data


def get_simulated_data_from_network(input_params, init_conditions, model, t_max, t_samples, perturbance):
    """
    Function:
        Simulates the inverted pendulum system control with the neural network.

    Parameters:
        input_params (list): Physical constants for the differential ecuation.
        init_conditions (list): Initial 'theta' and 'vel' values.
        model (model.keras): The neural network model.
        t_max (float): Maximum time for simulation.
        t_samples (int): Number of time samples.
        perturbance (bool): Indicates the presence of a perturbance in the simulation.

    Returns:
        dataset (dict):
            't': time (array)
            'theta': angle (array)
            'vel': angular velocity (array)
            'acc': angular acceleration (array)
            'torque': control torque (array)
    """
    global PERTURBANCE_SIM_COUNT
    
    dt = t_max / (t_samples - 1)
    t = np.linspace(0, t_max, t_samples)

    # Initial Conditions
    theta, vel = init_conditions

    # Lists to save the results
    theta_list = []
    vel_list = []
    acc_list = []
    torque_list = []

    if perturbance == True and PERTURBANCE_SIM_COUNT < PERTURBANCE_SIM_LIMIT:
        t_impulse = random.uniform(0.0, t_max/2)
        PERTURBANCE_SIM_COUNT += 1
    else:
        t_impulse = t_max + 1

    for i in range(t_samples):
        t_seconds = (i/t_samples) * t_max
        magnitude = random.uniform(-100, 100)
        f_ext = impulse(dt, t_seconds, t_impulse, magnitude)
        
        # Simulate model
        torque = model_predict(model, theta, vel)
        acc = get_inv_pendulum_acceleration(input_params, theta, vel, torque, 0)

        theta_list.append(theta)
        vel_list.append(vel)
        acc_list.append(acc)
        torque_list.append(torque)
        
        # Solve system with small time step
        S_0 = [theta, vel]
        _, theta_sol, vel_sol, acc_sol = solve_inv_pendulum_model(input_params, torque, S_0, 0, dt, 2, f_ext)
        
        theta = theta_sol[-1]   # Gets the last calculated theta
        vel = vel_sol[-1]       # Gets the last calculated velocity
        acc = acc_sol[-1]       # Gets the last calculated acceleration
        
    sim_data = {
        't': t,
        'theta': np.array(theta_list),
        'vel': np.array(vel_list),
        'acc': np.array(acc_list),
        'torque': np.array(torque_list)
    }

    return sim_data


def generate_dataset(n_sims, data_path, option, model=None, backup=False):
    """
    Function:
        Creates a dataset with multiple simulations of the 
        inverted pendulum system control (PID).

    Parameters:
        n_sims (int): Number of simulations.
        data_path (str): Path of the directory to save the dataset.
        option (str): 'PID' or 'NETWORK' depending on the type of 
                      simulation to execute.
        model (model.keras): Neural network model in case 'NETWORK'
                             option is provided.
        backup (bool): Whether to save or not a backup of the previous
                       dataset before creating the new one.

    Retorna:
        data_path (str): Full path where the dataset files were saved.
        norm_df_dataset (pd.DataFrame): Normalized dataset.
    """
    global PERTURBANCE_SIM_COUNT
    
    if option == "PID":
        print("\n--> Generating Dataset ...")
    elif option == "NETWORK":
        print("\n--> Simulating Network ...")
    else:
        print(f"-E-: {option} is not a valid option. Use 'PID' or 'NETWORK'.")
        return 1
    
    PERTURBANCE_SIM_COUNT = 0
    input_params = DYNAMIC_INPUT_PARAMS
    target = TARGET_ANGLE
    pid_gains = [PID_KP, PID_KI, PID_KD]
    t_stop = SIM_T_STOP
    t_samples = SIM_T_SAMPLES
    
    create_directory(data_path, backup, overwrite=False)
    
    df_list = [] # List of DataFrames/simulations
    
    for i in range(n_sims):
        id = i + 1 # Simulation Number
        init_conditions = [random.uniform(-0.2, 0.2),   # angle [rad]
                           random.uniform(-0.5, 0.5)    # velocity [rad/s]
                           ]
        
        perturbance = random.choice([True, False])
        
        if option == "PID":
            data = get_simulated_data(input_params, target, init_conditions, pid_gains, t_stop, t_samples, perturbance)
        elif option == "NETWORK":
            data = get_simulated_data_from_network(input_params, init_conditions, model, t_stop, t_samples, perturbance)
            
        # DataFrame generation (to save as .csv)
        df = pd.DataFrame({
            't': data['t'],
            'theta': data['theta'],
            'vel': data['vel'],
            'acc': data['acc'],
            'torque': data['torque']
        })
        df["simulacion"] = id # Simulation Number
        df_list.append(df)
        
        # Plot generation (to save as .png)
        t = data['t']
        model_solutions = {
            'theta': data['theta'],
            'vel': data['vel'],
            'acc': data['acc'],
            'torque': data['torque']
        }
        
        # Saves plots as .png
        create_simple_graph(
            x_values = t,
            x_title = "Tiempo",
            y_values = data['theta'],
            y_title = "Theta",
            annotate_values = [],
            plot_type = "plot",
            show_plot = False,
            graph_title = "Inverted Pendulum Simulation (Theta)",
            image_name = f"inv_pen_sim_theta{id}",
            image_path = data_path,
        )
        
        create_multi_y_graph(
            x_values = t,
            x_title = "Tiempo",
            y_values_dict = model_solutions,
            plot_type = "plot",
            show_plot = False,
            graph_title = "Inverted Pendulum Simulation",
            image_name = f"inv_pen_sim_{id}",
            image_path = data_path,
        )
    
    # Builds full dataframe
    df_dataset = pd.concat(df_list, ignore_index=True)
    
    # Variables for data normalization
    max_df_vals = df_dataset.max(axis=0) # Max of each dataset column
    min_df_vals = df_dataset.min(axis=0) # Min of each dataset column
    
    # Saves variables for data normalization into config file
    save_norm_config(max_df_vals, min_df_vals)
    
    # Normalizes dataframe
    norm_df_dataset = process_df("normalize", df_dataset, "", max_df_vals, min_df_vals)
    
    # Saves data in .csv
    if option == "PID":
        file_path = DATASET_CSV_PATH
        norm_df_dataset.to_csv(file_path, index=False)
        print(f"\nDataset saved as: '{file_path}'")
    elif option == "NETWORK":
        file_path = NETWORK_SIM_CSV_PATH
        norm_df_dataset.to_csv(file_path, index=False)
        print(f"\nNetwork simulation saved as: '{file_path}'")
    print("")
    
    return data_path, norm_df_dataset


def load_dataset(csv_path):
    """
    Function:
        Loads a saved dataset.

    Parameters:
        csv_path (str): Path of the .csv file to load.

    Retorna:
        dataset_df (pd.DataFrame): Dataset values in the form of a DataFrame.
    """
    
    print(f"\n--> Loading dataset from .csv file:\n{csv_path}\n")
    dataset_df = pd.read_csv(csv_path)
    return dataset_df


##### TEST FUNCTIONS (For Developers) #####

def test_solve_inv_pendulum_model():
    input_params = DYNAMIC_INPUT_PARAMS
    torque = 5.0
    initial_state = [0.1, 0.0]  # angle [rad], angular velocity [rad/s]

    t_start = 0
    t_stop = 20
    t_samples = 100

    t, theta, vel, acc = solve_inv_pendulum_model(input_params, torque, initial_state, t_start, t_stop, t_samples)

    model_solutions = {
        "theta [rad]" : theta,
        "vel [m/s]" : vel,
        "acc [m/s^2]" : acc
    }

    create_multi_y_graph(
        x_values = t,
        x_title = "Tiempo",
        y_values_dict = model_solutions,
        plot_type = "plot",
        show_plot = True,
        graph_title = "Inv Pendulum Model",
        image_name = "inv_pendulum_model_test",
        image_path = RUN_AREA,
    )


def test_pid_control():
    # Random values for testing
    kp = PID_KP
    ki = PID_KI
    kd = PID_KD
    dt = 0.1
    target = 0.0
    input_val = 0.8
    prev_integral_error = 0.05
    prev_error = 0.1

    torque, integral_error, error_out = pid_control(kp, ki, kd, target, input_val, prev_integral_error, prev_error, dt)

    print("Torque: ", torque)
    print("integral_error: ", integral_error)
    print("error_out: ", error_out)


def test_get_pid_gains():
    input_params = DYNAMIC_INPUT_PARAMS
    init_conditions = [0.2, 0.0]  # angle [rad], angular velocity [rad/s]
    get_pid_gains(input_params, init_conditions)


def test_get_simulated_data(option):
    input_params = DYNAMIC_INPUT_PARAMS
    initial_state = [0.1, 0.0] # angle [rad], angular velocity [rad/s]
    target = TARGET_ANGLE
    pid_gains = [PID_KP, PID_KI, PID_KD]
    t_stop = 10
    t_samples = 200
    
    if option == "PID":   
        data = get_simulated_data(input_params, target, initial_state, pid_gains, t_stop, t_samples, False)
    elif option == "NETWORK":
        model = load_model(MODEL_SAVE_PATH)
        data = get_simulated_data_from_network(input_params, initial_state, model, t_stop, t_samples, False)
    
    t = data['t']
    
    model_solutions = {
        'theta': data['theta'],
        'vel': data['vel'],
        'acc': data['acc'],
        'torque': data['torque']
    }

    create_simple_graph(
        x_values = t,
        x_title = "Tiempo",
        y_values = data['theta'],
        y_title = "Theta",
        annotate_values = [],
        plot_type = "plot",
        show_plot = True,
        graph_title = "Theta Simulation",
        image_name = f"theta_simulation",
        image_path = RUN_AREA,
    )

    create_multi_y_graph(
        x_values = t,
        x_title = "Tiempo",
        y_values_dict = model_solutions,
        plot_type = "plot",
        show_plot = True,
        graph_title = "All Model Simulation",
        image_name = f"all_model_simulation_test",
        image_path = RUN_AREA,
    )

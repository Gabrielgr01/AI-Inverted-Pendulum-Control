##### IMPORTS #####

# Third party imports
import numpy as np
from scipy.integrate import odeint

# Built-in imports
import random

# Local imports
from .config import *
from .visualization import *


##### FUNCTIONS DEFINITION #####

def get_inv_pendulum_acceleration(input_params, theta, vel, torque_control):
    """Calcula aceleración angular."""
    
    m = input_params[0]
    l = input_params[1]
    g = input_params[2]
    B = input_params[3]
    f_ext = input_params[4]
    
    acc = -(B/m*l**2)*vel + (g/l)*np.sin(theta) - (1/m*l)*f_ext - torque_control
    return acc


def get_inv_pendulum_torque(input_params, theta, vel, acc):
    """Calcula el torque."""

    m = input_params[0]
    l = input_params[1]
    g = input_params[2]
    B = input_params[3]
    f_ext = input_params[4]
    
    torque = l*m*g*np.sin(theta) - f_ext*l - B*vel - m*l**2*acc
    return torque


def get_inv_pendulum_model(S, t, input_params, torque_control):
    """
    Function: 
        Defines the differential equations for a inverted pendulum system.

    Parameters:
        S (list): State vector / Initial conditions [angle 'theta', velocity 'vel'].
        t (float): Time variable (not used explicitly but required by odeint).
        input_params (list): Physical constants for the differential ecuation.
        torque_control (float): System control torque.

    Returns:
        List: Derivatives [dtheta/dt, dv/dt].
    """
       
    theta, vel = S
    return [vel, get_inv_pendulum_acceleration(input_params, theta, vel, torque_control)]


def solve_inv_pendulum_model(input_params, torque_control, initial_state, t_start, t_stop, t_samples):
    """
    Function: 
        Solve the inverted pendulum system using the given parameters.

    Parameters:
        input_params (list): List with constants: k (stiffness) and b (cushioning).
        t_max (float): Maximum time for simulation.
        t_samples (int): Number of time samples.
        u (float): External force applied.

    Returns:
        List: time (t), angle (theta), velocity (v), and acceleration a.
    """
    
    S_0 = initial_state
    t = np.linspace(t_start, t_stop, t_samples)
    
    solution = odeint(get_inv_pendulum_model, y0=S_0, t=t, args=(input_params, torque_control))
    theta_sol = solution.T[0]
    vel_sol = solution.T[1]
    acc_sol = get_inv_pendulum_acceleration(input_params, theta_sol, vel_sol, torque_control)

    return t, theta_sol, vel_sol, acc_sol


def pid_control(kp, ki, kd, target, inputs, prev_integral_error, prev_error, dt):
    """
    Calcula la salida de un controlador PID.

    Parámetros:
        Kp, Ki, Kd: ganancias del controlador PID
        target: valor deseado
        inputs: valor actual del sistema (ej. ángulo)
        prev_integral_error: suma acumulada del error (para la parte integral)
        prev_error: error en el paso anterior (para la parte derivativa)
        dt: paso de tiempo

    Devuelve:
        torque: salida de control
        integral_error: valor actualizado
        error: valor actualizado
    """
    error = target - inputs
    integral_error = prev_integral_error + error * dt
    derivative_error = (error - prev_error) / dt

    torque = (kp * error) + (ki * integral_error) + (kd * derivative_error)

    return torque, integral_error, error


def get_pid_gains(input_params, target, t_max, t_samples):

    kp_values = np.linspace(5, 50, 10)
    ki_values = np.linspace(0, 5, 5)
    kd_values = np.linspace(0, 10, 5)

    min_error = float('inf')
    best_gains = (0, 0, 0)

    # Estado inicial
    theta_init = random.uniform(-0.2, 0.2)
    vel_init = random.uniform(-0.5, 0.5)
    init_conditions = [theta_init, vel_init]

    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                #t, salida = simular_pendulo(Kp, Ki, Kd)
                pid_gains = [kp, ki, kd]
                data = get_simulated_data(input_params, target, init_conditions, pid_gains, t_max , t_samples, )
                #mse = mean_squared_error([setpoint] * len(salida), salida)
                sim_theta = data['theta']
                current_error = np.mean((sim_theta - target)**2) # Mean squared error
    
                if current_error < min_error:
                    min_error = current_error
                    best_gains = pid_gains.copy()
    
    print(f"Mejor combinación PID: Kp={best_gains[0]:.2f}, Ki={best_gains[1]:.2f}, Kd={best_gains[2]:.2f}")
    print(f"Error cuadrático medio: {min_error:.5f}")
    
    return best_gains, min_error


def get_simulated_data(input_params, target, init_conditions, pid_gains, t_max, t_samples):
    """
    Simula el péndulo invertido con controlador PID paso a paso y genera un dataset.

    Parámetros:
        input_params (list): parámetros físicos [m, l, g, B, f_ext]
        kp, ki, kd (float): ganancias PID
        setpoint (float): valor deseado del ángulo (rad)
        t_max (float): tiempo total de simulación (s)
        t_samples (int): número de muestras temporales

    Retorna:
        dataset (dict): diccionario con arrays para:
            - 't': tiempo
            - 'theta': ángulo
            - 'vel': velocidad angular
            - 'acc': aceleración angular
            - 'torque': torque aplicado por el PID
    """

    dt = t_max / (t_samples - 1)
    t = np.linspace(0, t_max, t_samples)

    # Initial Conditions
    theta, vel = init_conditions

    # Variables para el PID
    integral_error = 0.0
    prev_error = 0.0 

    kp = pid_gains[0]
    ki = pid_gains[1]
    kd = pid_gains[2]

    # Listas para almacenar resultados
    theta_list = []
    vel_list = []
    acc_list = []
    torque_list = []

    for i in range(t_samples):
        # Calcular torque de control con PID
        torque, integral_error, prev_error = pid_control(
            kp, ki, kd, target, theta, integral_error, prev_error, dt
        )

        # Guardar datos
        theta_list.append(theta)
        vel_list.append(vel)
        torque_list.append(torque)
        if i == 0:
            acc_list.append(0)
        else:
            acc_list.append(acc)

        # Resolver dinámica del sistema en un paso pequeño
        S_0 = [theta, vel]
        #t_span = [0, dt]
        #solution = odeint(get_inv_pendulum_model, S_0, t_span, args=(input_params, torque))
        _, theta_sol, vel_sol, acc_sol = solve_inv_pendulum_model(input_params, torque, S_0, 0, dt, 2)
        theta = theta_sol[-1]   # Gets the las calculated theta
        vel = vel_sol[-1]     # Gets the las calculated velocity
        acc = acc_sol[-1]     # Gets the las calculated acceleration

    dataset = {
        't': t,
        'theta': np.array(theta_list),
        'vel': np.array(vel_list),
        'acc': np.array(acc_list),
        'torque': np.array(torque_list)
    }

    return dataset


def generate_dataset(n_simulations, input_params, t_max, t_samples, target_range=(-0.2, 0.2)):
    """
    Genera múltiples datasets de simulación del péndulo invertido.

    Parámetros:
        n_simulations (int): Número de simulaciones a realizar
        input_params (list): Parámetros físicos del sistema
        t_max (float): Tiempo total de cada simulación
        t_samples (int): Número de muestras por simulación
        target_range (tuple): Rango de valores objetivo (setpoint) para el PID

    Retorna:
        lista_datasets (list): Lista de diccionarios con los datos de cada simulación
    """
    dataset = []

    for _ in range(n_simulations):
        target = random.uniform(*target_range)
        sim_data = get_simulated_data(input_params, target, t_max, t_samples)
        #sim_data['target'] = target
        dataset.append(sim_data)

    return dataset


########## TESTS

def test_solve_inv_pendulum_model():
    # Parámetros físicos: m, l, g, B, f_ext
    input_params = [1.0, 1.0, 9.81, 0.1, 0.0]  # masa, largo, gravedad, fricción, fuerza externa

    # Torque de prueba (constante)
    torque = 0.5

    # Condiciones iniciales: ángulo y velocidad angular
    initial_state = [0.1, 0.0]  # 0.1 rad, 0 rad/s

    # Tiempo de simulación
    t_start = 0
    t_stop = 1
    t_samples = 100

    # Ejecutar simulación
    t, theta, vel, acc = solve_inv_pendulum_model(input_params, torque, initial_state, t_start, t_stop, t_samples)

    #print("t: ", t)
    #print("theta: ", theta)
    #print("vel: ", vel)
    #print("acc: ", acc)

    model_solutions = {
        "theta" : theta,
        "vel" : vel,
        "acc" : acc
    }

    create_multi_y_graph(
        x_values = t,
        x_title = "Tiempo",
        y_values_dict = model_solutions,
        plot_type = "scatter",
        show_plot = True,
        graph_title = "Inv Pendulum Model",
        image_name = "inv_pendulum_model_test",
        image_path = RUN_AREA,
    )

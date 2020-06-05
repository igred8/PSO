from objective_functions import rastrigin_function, ackley_path_function
from utilities import harness, runner, summarize
import numpy as np
import time, sys
sys.path.append('/home/vagrant/jupyter/repos/PSO')
from psolib import ImplicitTargetPSO
from parallel_psolib import SynchronousParallelPSO


domain = {
    'rastrigin': [[-5., 5.]],
    'ackley': [[-3., 9.]]
}


def variable_rastrigin(*args, **kwargs):
    sleep_time = np.random.uniform(0.0, 0.01)
    time.sleep(sleep_time)
    return rastrigin_function(*args, **kwargs)

def variable_ackley(*args, **kwargs):
    sleep_time = np.random.uniform(0.0, 0.01)
    time.sleep(sleep_time)
    return ackley_path_function(*args, **kwargs)


run_settings_rastrigin = {
    'function': variable_rastrigin,
    'searchspace': domain['rastrigin'],
    'target': None,  # Not used by ImplicitTargetPSO
    'nparticles': 50,
    'maxiter': 500,
    'precision': 1e-4,  # assume f(x_min) = 0
    'domain': domain['rastrigin'],
    'verbose': True
}

run_settings_ackley = {
    'function': variable_ackley,
    'searchspace': domain['ackley'],
    'target': None,  # Not used by ImplicitTargetPSO
    'nparticles': 50,
    'maxiter': 500,
    'precision': 1e-4,  # assume f(x_min) = 0
    'domain': domain['ackley'],
    'verbose': True
}

# Small test set
run_settings = run_settings_ackley
test_dimensions = [2, 4, 8, 16]
runs = 10
results = runner(SynchronousParallelPSO, run_settings, test_dimensions, runs)
summarize(results, run_settings, 'Ackley')
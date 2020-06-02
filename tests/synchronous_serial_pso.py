from objective_functions import rastrigin_function
import numpy as np
import time
from pathos.multiprocessing import ProcessingPool
from psolib import ImplicitTargetPSO


# Initial settings
DIM = 6

domain = {
    'rastrigin': np.repeat([[-5., 5]], DIM, axis=0)
}

run_settings = {
    'function': rastrigin_function,
    'searchspace': domain['rastrigin'],
    'target': None,  # Not used by ImplicitTargetPSO
    'nparticles': 50,
    'maxiter': 500,
    'precision': 1e-4,  # assume f(x_min) = 0
    'domain': domain['rastrigin'],
    'verbose': False
}


def harness(implementation, settings):
    # Must explicitly set random seed or every Pool result will be the same
    np.random.seed()
    opt = implementation()
    start = time.time()
    xarr, varr, parr, cparr, garr, cgarr = opt.run_pso(**settings)
    stop = time.time()
    best = cgarr[-1]
    iterations = cgarr.shape[0]
    timer = stop - start

    return best, iterations, timer

# RUNS = 25
# results = ProcessingPool().map(harnass, [ImplicitTargetPSO]*RUNS, [run_settings]*RUNS)


def rastrigin_runner():
    DIMS = [2, 4, 8, 16]
    RUNS = 25

    results = {}
    for DIM in DIMS:
        domain = {
            'rastrigin': np.repeat([[-5., 5]], DIM, axis=0)
        }
        run_settings['searchspace'] = domain['rastrigin']
        run_settings['domain'] = domain['rastrigin']
        result = ProcessingPool().map(harness, [ImplicitTargetPSO] * RUNS, [run_settings] * RUNS)
        results[DIM] = np.array(result)

    return results


def summarize(results, run_settings, func_name):
    akey = next(iter(results))
    print("For {function} each problem was run {runs} times\n".format(function=func_name, runs=results[akey].shape[0]))
    for dim, result in results.items():
        print("Input dimension: {dim}".format(dim=dim))
        print("Absolute solution tolerance: {}".format(run_settings['precision']))
        success = np.sum(result[:, 0] < run_settings['precision'])
        fraction = success / result.shape[0]
        std = np.std(result[:, 0])
        print("{:.2f} were successful. Standard Dev of best results: {:.3f}".format(fraction, std))
        runtime = np.sum(result[:, 2])
        avg_runtime = np.average(result[:, 2])
        print("Total run time: {:.1f}. Average run time: {:.1f".format(runtime, avg_runtime))


results = rastrigin_runner()
summarize(results, run_settings, 'Rastrigin')
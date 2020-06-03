from objective_functions import rastrigin_function
import numpy as np
import time
from psolib import ImplicitTargetPSO
from parallel_psolib import AsynchronousParallelPSO


# Initial settings
DIM = 2

domain = {
    'rastrigin': np.repeat([[-5., 5]], DIM, axis=0)
}


def variable_rastrigin(*args, **kwargs):
    sleep_time = np.random.uniform(0.0, 0.01)
    time.sleep(sleep_time)
    return rastrigin_function(*args, **kwargs)


run_settings = {
    'function': variable_rastrigin,
    'searchspace': domain['rastrigin'],
    'target': None,  # Not used by ImplicitTargetPSO
    'nparticles': 50,
    'maxiter': 500,
    'precision': 1e-4,  # assume f(x_min) = 0
    'domain': domain['rastrigin'],
    'verbose': True
}


def harness(implementation, settings):
    # Must explicitly set random seed or every Pool result will be the same
    np.random.seed()
    opt = implementation()
    start = time.time()
    xarr, varr, parr, cparr, garr, cgarr = opt.run_pso(**settings)
    stop = time.time()
    best = np.min(cgarr)
    iterations = cgarr.shape[0]
    timer = stop - start

    return best, iterations, timer


# ttt = 2
# if ttt == 1:
#     results = harness(ImplicitTargetPSO, run_settings)
# else:
#     results = harness(AsynchronousParallelPSO, run_settings)
# print(results)


def rastrigin_runner():
    DIMS = [2, 4, 8, 16]
    RUNS = 2

    results = {}

    for DIM in DIMS:
        result_data = []
        domain = {
            'rastrigin': np.repeat([[-5., 5]], DIM, axis=0)
        }
        run_settings['searchspace'] = domain['rastrigin']
        run_settings['domain'] = domain['rastrigin']
        for _ in range(RUNS):
            result = harness(AsynchronousParallelPSO, run_settings)
            print(result)
            result_data.append(result)
        results[DIM] = np.array(result_data)

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
        print("Total run time: {:.1f}. Average run time: {:.1f}".format(runtime, avg_runtime))


results = rastrigin_runner()
summarize(results, run_settings, 'Rastrigin')
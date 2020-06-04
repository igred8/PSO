import numpy as np
import time

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

    return best, iterations, timer, cgarr

def runner(implementation, run_settings, test_dimensions, runs):
    DIMS = test_dimensions  # [2, 4, 8, 16]
    RUNS = runs  # 2

    results = {}

    for DIM in DIMS:
        result_data = []
        domain = run_settings['domain'].copy()
        run_settings['domain'] = np.repeat(domain, DIM, axis=0)
        run_settings['searchspace'] = np.repeat(domain, DIM, axis=0)
        for i in range(RUNS):
            result = harness(implementation, run_settings)
            print(result[:-1])
            result_data.append(result[:-1])
            np.save('{imp}_{func}_{dim}_{i}_cgarr.npy'.format(imp=implementation.__name__, 
                                                        func=run_settings['function'].__name__,
                                                        dim=DIM,
                                                        i=i), result[-1])
            np.save('{imp}_{func}_{dim}_{i}_summary.npy'.format(imp=implementation.__name__, 
                                                        func=run_settings['function'].__name__,
                                                        dim=DIM,
                                                        i=i), result[:-1])
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


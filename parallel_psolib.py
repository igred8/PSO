from psolib import ImplicitTargetPSO
import numpy as np
from pathos.multiprocessing import Pool, ProcessPool, cpu_count
import time


class SynchronousParallelPSO(ImplicitTargetPSO):

    def run_pso(self, function, searchspace, target, nparticles, maxiter, precision, domain,
                verbose=True, pool_size=None):

        """ Performs a PSO for the given function in the searchspace, looking for the target, which is in the output space.

        This is an extremely straightforward parallelization of the PSO algorithm as implemented in psolib. If performs
        a parallel evaluation of all particles through a multiprocessing pool and then updates the simulation state
        globally. This is as opposed to the typical algorithm which updates the global state after every particle
        evaluation. This makes this algorithm less efficient at finding minima. But it is very straightforward
        programmatically and is included as a comparison and stepping stone to an asynchronous update version.

        function - the function to be optimized. Its domain must include the seachspace and its output must be in the space of target.
        searchspace - np.array((ssdim, 2))
        target - Not used by `ImplicitTargetPSO`. `function` should include any necessary target data.
        nparticles - number of particles to use in the optimization
        maxiter - maximum number of iterations to the optimization routine
        precision - how close to the target to attemp to get
        domain - absolute boundaries on the trial solutions/particles
        pool_size - (int) set the ProcessingPool size explicitly. Defaults to 4 if not set.
        """
        if not pool_size:
            pool_size = 4

        # update attributes
        self.maxiter = maxiter
        self.precision = precision

        # search space dimensionality
        if searchspace.shape[1] != 2:
            print('WARNING! searchspace does not have dimenstions (N,2).')
        ssdim = searchspace.shape[0]

        # init particle positions and velocities
        xpart = np.random.random((nparticles, ssdim))

        for ii in range(ssdim):
            xpart[:, ii] = (searchspace[ii, 1] - searchspace[ii, 0]) * xpart[:, ii] + searchspace[
                ii, 0]  # scale the uniform radnom dist

        vpart = np.zeros(xpart.shape)

        # init particle best solution
        pbest = 1.0 * xpart
        # NOTE: Best not to assume the form of obj function input
        cpbest = np.array([self.cost(function(*xp), target) for xp in pbest])
        # init global best solutions
        im = np.argmin(cpbest)
        gbest = pbest[im]
        cgbest = cpbest[im]

        if False:
            return xpart, vpart, pbest, cpbest, gbest, cgbest

        # intermediate arrays
        # multiply by 1.0 to make copies not bind references
        xarr = 1.0 * xpart[:, :, None]
        varr = 1.0 * vpart[:, :, None]
        parr = 1.0 * pbest[:, :, None]
        cparr = 1.0 * cpbest[:, None]
        garr = 1.0 * gbest[:, None]
        cgarr = 1.0 * np.array([cgbest])

        iternum = 0
        pool = Pool(pool_size)

        t1 = time.time()
        while (iternum <= maxiter) and (cgbest > precision):

            # update velocity
            vpart = self.velocity(vpart, xpart, pbest, gbest)
            # update position
            xpart = xpart + vpart

            # keeps particles inside the absolute boundaries given by `domain`
            xpart = np.maximum(xpart, domain[:, 0])
            xpart = np.minimum(xpart, domain[:, 1])

            # compute cost of new position
            # We completely drop cost since it is not needed
            cpp = pool.map(function, xpart)
            cpp = np.array(cpp)

            # if iternum == 0:
            #     print('Initial cpp', cpp)
            # update best position
            cpp_better = cpp < cpbest
            pbest[cpp_better] = xpart[cpp_better]
            cpbest[cpp_better] = cpp[cpp_better]

            best_id = np.argmin(cpp)
            gbest = xpart[best_id]
            cgbest = cpp[best_id]


            xarr = np.concatenate((xarr, xpart[:, :, None]), axis=2)
            varr = np.concatenate((varr, vpart[:, :, None]), axis=2)
            parr = np.concatenate((parr, pbest[:, :, None]), axis=2)
            cparr = np.concatenate((cparr, cpbest[:, None]), axis=1)
            garr = np.concatenate((garr, gbest[:, None]), axis=1)
            cgarr = np.append(cgarr, cgbest)

            iternum += 1

        t2 = time.time()
        if verbose:
            print('optimization took {:5.2f} seconds'.format(*[t2 - t1]))

        return xarr, varr, parr, cparr, garr, cgarr


class AsynchronousParallelPSO(ImplicitTargetPSO):

    def run_pso(self, function, searchspace, target, nparticles, maxiter, precision, domain,
                verbose=True, pool_size=None):

        """ Performs a PSO for the given function in the searchspace, looking for the target, which is in the output space.

        The asynchronous evaluation means the exact definition of iterations may be lost. To preserve some sense of this
        an iteration is defined to be `nparticles` evaluations performed. This means that not every particle is updated
        in the history for every iteration. However, the total number of function evaluations (iterations * nparticles)
        will still be preserved for this definition.

        function - the function to be optimized. Its domain must include the seachspace and its output must be in the space of target.
        searchspace - np.array((ssdim, 2))
        target - Not used by `ImplicitTargetPSO`. `function` should include any necessary target data.
        nparticles - number of particles to use in the optimization
        maxiter - maximum number of iterations to the optimization routine
        precision - how close to the target to attemp to get
        domain - absolute boundaries on the trial solutions/particles
        pool_size - (int) set the ProcessingPool size explicitly. Defaults to 4 if not set.
        """
        if not pool_size:
            pool_size = 4

        # update attributes
        self.maxiter = maxiter
        self.precision = precision

        # search space dimensionality
        if searchspace.shape[1] != 2:
            print('WARNING! searchspace does not have dimenstions (N,2).')
        ssdim = searchspace.shape[0]

        # init particle positions and velocities
        xpart = np.random.random((nparticles, ssdim))

        for ii in range(ssdim):
            xpart[:, ii] = (searchspace[ii, 1] - searchspace[ii, 0]) * xpart[:, ii] + searchspace[
                ii, 0]  # scale the uniform radnom dist

        vpart = np.zeros(xpart.shape)

        # init particle best solution
        pbest = 1.0 * xpart
        # NOTE: Best not to assume the form of obj function input
        cpbest = np.array([self.cost(function(*xp), target) for xp in pbest])
        # init global best solutions
        im = np.argmin(cpbest)
        gbest = pbest[im]
        cgbest = cpbest[im]

        if False:
            return xpart, vpart, pbest, cpbest, gbest, cgbest

        # intermediate arrays
        # multiply by 1.0 to make copies not bind references
        xarr = 1.0 * xpart[:, :, None]
        varr = 1.0 * vpart[:, :, None]
        parr = 1.0 * pbest[:, :, None]
        cparr = 1.0 * cpbest[:, None]
        garr = 1.0 * gbest[:, None]
        cgarr = 1.0 * np.array([cgbest])

        iternum = 0
        evalnum = 0
        # Asynchronous process management
        pool = ProcessPool(pool_size)
        results = []

        # initial submission

        for fi in range(nparticles):
            # update velocity
            vpart[fi] = self.velocity(vpart[fi], xpart[fi], pbest[fi], gbest)
            # update position
            xpart[fi] = xpart[fi] + vpart[fi]

            # keeps particles inside the absolute boundaries given by `domain`
            xpart[fi] = np.maximum(xpart[fi], domain[:, 0])
            xpart[fi] = np.minimum(xpart[fi], domain[:, 1])

            # compute cost of new position
            results.append(pool.apipe(function, xpart[fi]))

        t1 = time.time()
        while (iternum <= maxiter) and (cgbest > precision):

            for i, res in enumerate(results):
                if res.ready():
                    # Get result and update
                    cpp = res.get()

                    # update best position
                    if cpp < cpbest[i]:
                        pbest[i] = xpart[i]
                        cpbest[i] = cpp
                    if cpp < cgbest:
                        gbest = xpart[i]
                        cgbest = cpp

                    # update velocity
                    vpart[i] = self.velocity(vpart[i], xpart[i], pbest[i], gbest)
                    # update position
                    xpart[i] = xpart[i] + vpart[i]

                    # keeps particles inside the absolute boundaries given by `domain`
                    xpart[i] = np.maximum(xpart[i], domain[:, 0])
                    xpart[i] = np.minimum(xpart[i], domain[:, 1])

                    # Resubmit
                    results[i] = pool.apipe(function, xpart[i])

                    evalnum += 1

            current_iternum = evalnum // nparticles

            if (current_iternum > iternum) or (cgbest < precision):

                xarr = np.concatenate((xarr, xpart[:, :, None]), axis=2)
                varr = np.concatenate((varr, vpart[:, :, None]), axis=2)
                parr = np.concatenate((parr, pbest[:, :, None]), axis=2)
                cparr = np.concatenate((cparr, cpbest[:, None]), axis=1)
                garr = np.concatenate((garr, gbest[:, None]), axis=1)
                cgarr = np.append(cgarr, cgbest)

                iternum = current_iternum

        t2 = time.time()
        if verbose:
            print('optimization took {:5.2f} seconds'.format(*[t2 - t1]))

        return xarr, varr, parr, cparr, garr, cgarr
import time
import random


from roadrunner import RoadRunner, Logger
import numpy as n
import os
import sys
import random
from multiprocessing import Process, Queue, cpu_count

def test():
    src = '/Users/andy/Library/Python/2.7/lib/python/site-packages/roadrunner/testing/dsmts/dsmts-003-01.xml'
    simArgs = (0, 100, 50)
    simKWArgs = {'integrator':'gillespie', 'reset':True}

    # need one less because of queue
    n =  2**15 - cpu_count() - 2


    return ensemble(src, n, None, *simArgs, **simKWArgs)


def ensemble(src, ensembles, seeds=None, *sim_args, **sim_kwargs):
    """Run an ensemble simulation in parallel. 

    Args:
        src: an sbml string or file name.

        ensembles: how many ensembles to run, at the moment, limited to 2**15

        seeds: an optional list of seeds to use. If not given, random seeds
               are assigned to each simulation. 

        *args: the arguments that are passed to RoadRunner.simulate
    
        **kwargs: the keyword arguments that are passed to RoadRunner.simulate

    Returns:
        A tuple containing the mean and std matricies. 

    """

    import time

    start = time.time()

    # parent process will run statistics, but stats are
    # very minimal, so better perf using all processors.
    CPU_COUNT = cpu_count() 

    # the multiprocessing queue has a limit of 2^15 
    # on many platforms.
    MAX_SIZE = 2**15 - CPU_COUNT


    # Create queues
    seedQ = Queue()
    results = Queue()

    # Submit tasks
    print("creating {} tasks".format(ensembles))
    
    if seeds is None:
        for _ in xrange(ensembles):
            seedQ.put(random.randint(0, 2**31 - 1))
    else:
        for i in xrange(ensembles):
            seedQ.put(seeds[i])
        

    #insert setinals
    for i in xrange(CPU_COUNT):
        seedQ.put(None)
        

    print("creating worker processes...")

    # Start worker processes
    for i in range(CPU_COUNT):
        Process(target=_worker, \
                args=(src, seedQ, results, sim_args, sim_kwargs)).start()

    print('done creating  processes')

    # stdev and mean matricies
    # need to wait until first result to know size
    stdev = None
    mean = None

    # Get and print results
    print('Unordered results:')
    for i in range(ensembles):

        res = results.get()

        if stdev is None or mean is None:
            mean = n.zeros(res.shape)
            stdev = n.zeros(res.shape)  

            # time column
            mean[:,0] = res[:,0]
            stdev[:,0] = res[:,0]

        # add the data columns to mean
        mean[:,1:] += res[:,1:]

        # square the result data, and add to 
        # stdev, we want sqrt(|x^2| - |x|^2)
        stdev[:,1:] += (res[:,1:])**2


    # divide by num processes to get expectation of x
    # and expectation of X^2
    mean[:,1:] = mean[:,1:] / ensembles
    stdev[:,1:] = stdev[:,1:] / ensembles

    stdev [:,1:] = n.sqrt(stdev[:,1:] - (mean[:,1:])**2)


    # all done with simulations
    print("performed {} simulations in {} seconds".format(ensembles, time.time() - start))

    import matplotlib.pyplot as p


    time = mean[:,0]

    for i in xrange(1, mean.shape[1]):
        print("i: " + str(i))
        p.plot(time, mean[:,i])
        p.errorbar(time, mean[:,i], yerr=stdev[:,i], marker='o')


    p.show()

    return (mean, stdev)
        
#
# Function run by worker processes
#
def _worker(src, seeds, results, sim_args, sim_kwargs):
    # create a rr obj

    print("starting worker processes: " + str(os.getpid()))

    n = 0;

          
    r = RoadRunner(src);

    # remove and apply the selections 
    # no sense in re-evaluating selection list each time.
    if 'sel' in sim_kwargs:
        r.selections = sim_kwargs['sel']
        del sim_kwargs['sel']

    if 'selections' in sim_kwargs:
        r.selections = sim_kwargs['selections']
        del sim_kwargs['selections']


    # might have forgot to reset sim, 
    # default is to reset sim between each simulation.
    if not 'reset' in sim_kwargs:
        Logger.log(Logger.LOG_WARNING, 
                   "simulate args missing reset, defaulting to reset between simulations")
        sim_kwargs['reset'] = True

    # read until None, 
    # blocks if queue is empty
    for seed in iter(seeds.get, None):
        sim_kwargs['seed'] = seed
        result = r.simulate(*sim_args, **sim_kwargs)
        results.put(result)
        n += 1

    print("process {} finished with {} simulations".format(os.getpid(), n))





if __name__ == '__main__':
    mean, stdev = test()

    print("mean: ")
    print(mean)

    print("stdev: ")
    print(stdev)


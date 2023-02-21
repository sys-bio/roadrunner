import numpy as np
from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
from threading import Thread
from multiprocessing import Queue
import time
from platform import platform
import cpuinfo # pip install py-cpuinfo

NTHREADS = 16
NTHREADS = 128
NSIMS = 100000
NSTEPS = 11


class SimulationWorkerThread(Thread):

    def __init__(self, input_queue: Queue, results_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.results_queue = results_queue

    def run(self) -> None:
        for rr in iter(self.input_queue.get, "STOP"):
            assert isinstance(rr, RoadRunner)
            rr.resetAll()
            self.results_queue.put(rr.simulate(0, 10, NSTEPS))

if __name__ == '__main__':
    # setup timing
    start = time.time()

    # get sbml to work with from one of our test modules
    sbml = tmf.BatchImmigrationDeath03().str()

    # create our roadrunner instance
    rr = RoadRunner(sbml)

    # set up a stochastic simulation
    rr.setIntegrator('gillespie')

    # set the seed for reproducuble example
    gillespie_integrator = rr.getIntegrator()
    gillespie_integrator.seed = 1234

    input_queue = Queue()
    results_queue = Queue()

    # populate input queue
    for i in range(NSIMS):
        input_queue.put(rr)

    # create threads
    for _ in range(NTHREADS):
        SimulationWorkerThread(input_queue, results_queue).start()

    # get results
    data = np.ndarray((NSIMS, NSTEPS, 2))
    for i in range(NSIMS):
        data[i] = results_queue.get()

    # terminate the continually iterating threads
    for _ in range(NTHREADS):
        input_queue.put("STOP")

    duration = time.time() - start

    # the time it took in serial
    serial_time = 64.92753291130066

    # compute speedup
    speedup = serial_time / duration

    print(f'Took {duration} seconds to run', NSIMS, 'stochastic simulations with', NTHREADS, 'threads')
    print(f'Speed up is {speedup}')
    cpu_info = cpuinfo.get_cpu_info()
    print(f'Platform: {platform()}')
    print('python_version:', cpu_info['python_version'])
    print('Processor:', cpu_info['brand_raw'])

    '''
    Output: 
    Took 38.20730757713318 seconds to run 100000 stochastic simulations with 16 thread
    Speed up is 1.6993485547293408
    Platform: Windows-10-10.0.22000-SP0
    python_version: 3.9.5.final.0 (64 bit)
    Processor: 11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz
    '''

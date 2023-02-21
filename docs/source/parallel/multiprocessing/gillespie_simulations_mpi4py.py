import numpy as np
from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
from threading import Thread
from multiprocessing import Queue
import time
from platform import platform
import cpuinfo  # pip install py-cpuinfo
import mpi4py   # pip install mpi4py

NTHREADS = 16
NSIMS = 100000
NSTEPS = 11










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

    # the time it took in serial
    serial_time = 64.92753291130066












    # compute speedup
    # speedup = serial_time / duration
    #
    # print(f'Took {duration} seconds to run', NSIMS, 'stochastic simulations with', NTHREADS, 'threads')
    # print(f'Speed up is {speedup}')
    # cpu_info = cpuinfo.get_cpu_info()
    # print(f'Platform: {platform()}')
    # print('python_version:', cpu_info['python_version'])
    # print('Processor:', cpu_info['brand_raw'])
    #
    # '''
    # Output:
    # Took 38.20730757713318 seconds to run 100000 stochastic simulations with 16 thread
    # Speed up is 1.6993485547293408
    # Platform: Windows-10-10.0.22000-SP0
    # python_version: 3.9.5.final.0 (64 bit)
    # Processor: 11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz
    # '''












import sys
sys.path += [
    r'D:\roadrunner\roadrunner\roadrunner-install-rel-llvm6\site-packages',
    # r'D:\roadrunner\roadrunner\roadrunner-install-rel-llvm12\site-packages'
]

import roadrunner
assert "llvm6" in roadrunner.__file__, roadrunner.__file__

from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
import time
import numpy as np
from platform import platform
import cpuinfo # pip install py-cpuinfo



NSIMS = 1000000

if __name__ == '__main__':
    # setup timing
    start = time.time()

    # get sbml to work with from one of our test modules
    sbml = tmf.BatchImmigrationDeath03().str()

    # create our roadrunner instance
    rr = RoadRunner(sbml)

    # set up a stochastic simulation
    rr.setIntegrator('gillespie')

    # set the seed for reproducible example
    gillespie_integrator = rr.getIntegrator()
    gillespie_integrator.seed = 1234

    start_time = 0
    end_time = 10
    num_points = 11

    # preallocate for efficiency
    data = np.ndarray((NSIMS, num_points, 2))
    for simulation_number in range(NSIMS):
        rr.resetAll()
        data[simulation_number] = rr.simulate(start_time, end_time, num_points)

    print(data)
    print(data.shape)

    print('Took',  time.time() - start, 'seconds to run', NSIMS, 'stochastic simulations on 1 core')
    cpu_info = cpuinfo.get_cpu_info()
    print(f'Platform: {platform()}')
    print('python_version:', cpu_info['python_version'])
    print('Processor:', cpu_info['brand_raw'])

    '''
    Output:
        Took 64.92753291130066 seconds to run 1000000 stochastic simulations on 1 core

        Platform: Windows-10-10.0.22000-SP0
        python_version: 3.9.5.final.0 (64 bit)
        Processor: 11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz
    '''

import numpy as np

from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
from multiprocessing import cpu_count
import ray
import time
from platform import platform
import cpuinfo  # pip install py-cpuinfo

NCORES = cpu_count()
NSIMS = 1000000

ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActorPath(object):
    """Ray actor to execute simulations."""

    def __init__(self, rr: RoadRunner):
        self.rr: RoadRunner = rr

    def simulate(self, size=1):
        num_points = 101
        results = np.ndarray((size, num_points, 2))  # 2 for 1 model species and time
        for k in range(size):
            self.rr.resetAll()
            results[k] = self.rr.simulate(0, 100, num_points)
        return results


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

    simulators = [SimulatorActorPath.remote(rr) for _ in range(NCORES)]

    # run simulations
    tc_ids = []
    for k, simulator in enumerate(simulators):
        tcs_id = simulator.simulate.remote(size=int(np.floor(NSIMS / NCORES)))
        tc_ids.append(tcs_id)
    results = ray.get(tc_ids)
    print(results)

    duration = time.time() - start

    # the time it took in serial
    serial_time = 64.92753291130066

    # compute speedup
    speedup = serial_time / duration

    print(f'Took {duration} seconds to run', NSIMS, 'stochastic simulations on', NCORES, 'cores')
    print(f'Speed up is {speedup}')
    cpu_info = cpuinfo.get_cpu_info()
    print(f'Platform: {platform()}')
    print('python_version:', cpu_info['python_version'])
    print('Processor:', cpu_info['brand_raw'])

    '''
    Output: 
        Took 99.32935857772827 seconds to run 1000000 stochastic simulations on 16 cores
        Speed up is 0.6536590373780867
        Platform: Windows-10-10.0.22000-SP0
        python_version: 3.9.5.final.0 (64 bit)
        Processor: 11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz
    '''

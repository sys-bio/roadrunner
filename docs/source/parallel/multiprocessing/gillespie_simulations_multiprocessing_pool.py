from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
from multiprocessing import Pool, cpu_count
import time
from platform import platform
import cpuinfo # pip install py-cpuinfo

NCORES = cpu_count()
NSIMS = 1000000

def simulate_worker(rr: RoadRunner):
    rr.resetAll()
    return rr.simulate(0, 10, 11)


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

    # create a processing pool
    p = Pool(processes=NCORES)

    # perform the simulations
    arrays = p.map(simulate_worker, [rr for i in range(NSIMS)])

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
        Took 19.231333017349243 seconds to run 1000000 stochastic simulations on 16 cores
        Speed up is 3.3761327336346008
        Platform: Windows-10-10.0.22000-SP0
        python_version: 3.9.5.final.0 (64 bit)
        Processor: 11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz
    '''

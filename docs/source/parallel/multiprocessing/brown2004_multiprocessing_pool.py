from roadrunner import RoadRunner
from roadrunner.tests import TestModelFactory as tmf
from multiprocessing import Pool, cpu_count
import time
from platform import platform
import cpuinfo # pip install py-cpuinfo
import numpy as np

from multiprocessing import Pool, cpu_count

NCORES = cpu_count()
NSIMS = 10000


def simulate_worker(rr: RoadRunner):
  rr.resetAll()
  rr.simulate(0, 10000, 10000)


def parallel(rr : RoadRunner, N: int):
  p = Pool(processes=cpu_count())
  p.map(simulate_worker, [rr for i in range(N)])


def serial(rr: RoadRunner, N: int):
  for i in range(N):
    rr.simulate(0, 10000, 10000)


if __name__ == '__main__':

  nsimsVec = [1, 10, 100, 1000, 10000, 100000]
  for i in nsimsVec:
    start = time.time()

    # rr = RoadRunner(tmf.SimpleFlux().str())
    rr = RoadRunner(tmf.Brown2004().str())
    # serial(rr, N=i)
    parallel(rr, N=i)

    duration = time.time() - start

    print(f'Took {duration} seconds to run', i, 'deterministic simulations on', NCORES, 'cores')
    cpu_info = cpuinfo.get_cpu_info()
    print(f'Platform: {platform()}')
    print('python_version:', cpu_info['python_version'])
    print('Processor:', cpu_info['brand_raw'])
    print()


from threading import Thread
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
import numpy as np
from multiprocessing import Pool
import unittest
import pandas as pd
import pickletools
import sys
import pickle
import os
import ray
import time
import copy

from os.path import dirname, exists, join

print(f"Python interpreter at: {sys.executable}")

# sys.path += [
#     # r'/mnt/d/roadrunner/roadrunner/cmake-build-release-wsl/lib/site-packages'
#     # r"D:\roadrunner\roadrunner\install-msvc2019-rel\site-packages",
#     # r"D:\roadrunner\roadrunner\cmake-build-release-visual-studio\lib\site-packages"
#
# ]
import roadrunner
from roadrunner import *

import roadrunner.testing.TestModelFactory as tmf
from roadrunner.roadrunner import RoadRunner


def simulate_return_None(r):
    r.resetAll()
    print(r.simulate(0, 10, 11))


def simulate_return_dataframe(r):
    r.resetAll()
    data = r.simulate(0, 10, 11)
    df = pd.DataFrame(data, columns=data.colnames)
    return df


def simulate_return_NamedArray(r):
    r.resetAll()
    return r.simulate(0, 10, 11)


@ray.remote
class SimulatorActorPath(object):
    """Ray actor to execute simulations."""

    def __init__(self, r: roadrunner.RoadRunner):
        self.r: roadrunner.RoadRunner = r

    def simulate(self, size=1):
        import roadrunner
        num_points = 101
        results = np.ndarray((size, num_points, 2))  # 2 for 1 model species and time
        for k in range(size):
            self.r.resetAll()
            results[k] = self.r.simulate(0, 100, num_points)
        return results


class RoadRunnerPickleTests(unittest.TestCase):

    def setUp(self):
        rr = RoadRunner(tmf.SimpleFlux().str())
        rr.setIntegrator("gillespie")
        gillespie = rr.getIntegrator()
        gillespie.seed = 123
        self.rr = rr

    def tearDown(self):
        pass

    def test_to_pickle_and_back(self):
        pfile = os.path.join(os.path.dirname(__file__), "pkl.serialization")
        with open(pfile, 'wb') as f:
            pickle.dump(self.rr, f)

        self.assertTrue(os.path.isfile(pfile))

        with open(pfile, "rb") as f:
            rr = pickle.load(f)
        try:
            data = rr.simulate(0, 10, 11)
        except Exception:
            self.fail("Cannot simulate a serialization loaded model")

        self.assertEqual(data.shape, (11, 3))

        if os.path.isfile(pfile):
            os.remove(pfile)

    def test_we_can_copy_rr(self):
        rr2 = copy.copy(self.rr)
        self.assertNotEqual(hex(id(self.rr)), hex(id(rr2)))

    def test_pool_returns_None(self):
        p = Pool(processes=4)
        p.map(simulate_return_None, [self.rr for i in range(10)])
        p.close()

    def test_pool_returns_DataFrame(self):
        N = 10
        p = Pool(processes=4)
        dfs = p.map(simulate_return_dataframe, [self.rr for i in range(N)])
        p.close()
        self.assertEqual(len(dfs), 10)
        dct = {i: df for i, df in enumerate(dfs)}
        df = pd.concat(dct, axis=0)
        print(df)
        self.assertEqual(df.shape, (N * 11, 3))

    def test_pool_returns_NamedArray(self):
        N = 10
        p = Pool(processes=4)
        arrs = p.map(simulate_return_NamedArray, [self.rr for i in range(N)])
        p.close()
        print(arrs)
        data = np.vstack(arrs)
        self.assertEqual((110, 3), data.shape)

    @unittest.skip("Works when you run on its own but "
                   "No module named 'roadrunner' error when "
                   "in test suite with other tests, on windows only. ")
    def test_using_ray_library(self):
        import roadrunner
        ray.init(ignore_reinit_error=True)

        actor_count = 10  # cores to run this on

        rr = roadrunner.RoadRunner(tmf.SimpleFlux().str())
        simulators = [SimulatorActorPath.remote(rr) for _ in range(actor_count)]

        # run simulations
        sim_per_actor = 10
        tc_ids = []
        for k, simulator in enumerate(simulators):
            tcs_id = simulator.simulate.remote(size=sim_per_actor)
            tc_ids.append(tcs_id)
        results = ray.get(tc_ids)
        data = np.vstack(results)
        self.assertEqual((100, 101, 3), data.shape)

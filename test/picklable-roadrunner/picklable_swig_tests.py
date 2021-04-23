import unittest

import sys
import pickle
import os
sys.path += [
    # r"D:\roadrunner\roadrunner\install-msvc2019-rel\site-packages",
    r"D:\roadrunner\roadrunner\cmake-build-release-visual-studio\lib\site-packages"
]

import roadrunner.testing.TestModelFactory as tmf
from roadrunner.roadrunner import RoadRunner


class RoadRunnerPickleTests(unittest.TestCase):
    ignore = [
                "<class 'method'>",
                "<class 'builtin_function_or_method'>",
                "<class 'method-wrapper'>",
                "<class 'function'>"
            ]
    def setUp(self):
        self.testModelObj = tmf.TestModelFactory("SimpleFlux")
        self.testModelSbml = self.testModelObj.str()

    def tearDown(self):
        pass

    def test(self):
        """
        S1 <class 'float'>
        S1_amt <class 'float'>
        S1_conc <class 'float'>
        S2 <class 'float'>
        S2_amt <class 'float'>
        S2_conc <class 'float'>
        _J0 <class 'float'>
        _J1 <class 'float'>
        __class__ <class 'type'>
        __dict__ <class 'dict'>
        __doc__ <class 'str'>
        __module__ <class 'str'>
        __weakref__ <class 'NoneType'>
        _properties <class 'list'>
        conservedMoietyAnalysis <class 'bool'>
        default_compartment <class 'float'>
        diffstep <class 'float'>
        kb <class 'float'>
        kf <class 'float'>
        selections <class 'list'>
        steadyStateSelections <class 'list'>
        steadyStateThresh <class 'float'>
        this <class 'SwigPyObject'>
        thisown <class 'bool'>
        timeCourseSelections <class 'list'>
        steadyStateSolver <class 'roadrunner.roadrunner.SteadyStateSolver'>
        options <class 'roadrunner.roadrunner.RoadRunnerOptions'>
        model <class 'roadrunner.roadrunner.ExecutableModel'>
        integrator <class 'roadrunner.roadrunner.Integrator'>
        _RoadRunner__simulateOptions <class 'roadrunner.roadrunner.SimulateOptions'>

        :return:
        """
        r = RoadRunner(self.testModelSbml)

        for i in sorted(dir(r)):
            t = type(getattr(r, i))
            if str(t) not in self.ignore:
                print(i, t)

    def test2(self):
        r1 = RoadRunner(self.testModelSbml)
        s = r1.getSteadyStateSolver()
        print(s.getSettingsMap())

        r1.setSteadyStateSolver("newton")
        s = r1.getSteadyStateSolver()
        print(type(s))
        print(s.getSettingsMap())
        print(s)
        fname = os.path.join(os.path.dirname(__file__), "pickletest.dat")
        with open(fname, "wb") as f:
            pickle.dump(s, f)
        print(fname)

        with open(fname, "rb") as f:
            sloaded = pickle.load(f)

        # print(sloaded)
        # for k, v in s.__class__.__dict__.items():
        #     print(k, v)
            # if str(v) in self.ignore:
            #     print(k, type(v)


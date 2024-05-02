# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:39:24 2020

@author: Lucian Smith
"""

from __future__ import division

import os.path
from os import walk, scandir
from os.path import isdir
import time
import platform

import sys

sys.path += [
    r"D:\roadrunner\roadrunner\cmake-build-release\lib\site-packages",
    r"/Users/ciaranwelsh/Documents/roadrunner/cmake-build-release-llvm-13/lib/site-packages"
]

import csv
import unittest
import numpy as np
import roadrunner


class RoadRunnerTests(unittest.TestCase):
    stochdir = None
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.stochdir = ""
        for dir in ["../sbml-test-suite/stochastic/", "../test/sbml-test-suite/stochastic/",
                    "../../test/sbml-test-suite/stochastic/", "../../../test/sbml-test-suite/stochastic/",
                    "../../../../test/sbml-test-suite/stochastic/", "../../../../../test/sbml-test-suite/stochastic/",
                    "../../../../../../test/sbml-test-suite/stochastic/",
                    "../roadrunner/sbml-test-suite/stochastic/", "../roadrunner/test/sbml-test-suite/stochastic/",
                    "../../roadrunner/test/sbml-test-suite/stochastic/",
                    "../../../roadrunner/test/sbml-test-suite/stochastic/",
                    "../../../../roadrunner/test/sbml-test-suite/stochastic/",
                    "../../../../../roadrunner/test/sbml-test-suite/stochastic/",
                    "../../../../../../roadrunner/test/sbml-test-suite/stochastic/"]:
            if isdir(dir):
                cls.stochdir = dir
        if cls.stochdir == "":
            raise ValueError("Unable to find stochastic test suite directory.")
        cls.outfile = "results.tsv"
        cls.nrepeats = 10000

        cls.onlysometests = False
        cls.sometests = ["00001", "00040"]

        cls.stochdirs = [f.path for f in scandir(cls.stochdir) if f.is_dir()]

        cls.results = open(cls.stochdir + cls.outfile, "w")
        cls.results.write("Test #")
        cls.results.write("\tLevel/Version")
        cls.results.write("\tSynopsis")
        cls.results.write("\tnWrongMeans")
        cls.results.write("\tnWrongSDs")
        cls.results.write("\tnWrongLogMeans")
        cls.results.write("\tnWrongLogSDs")
        cls.results.write("\n")

        print(platform.uname())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results.close()

    def countWrongMeans(self, means, expectedmeans, expectedsds, xrange):
        nmean_wrong = 0
        Zvec = np.sqrt(self.nrepeats) * (means - expectedmeans) / expectedsds
        for Z in Zvec:
            if Z < xrange[0] or Z > xrange[1]:
                nmean_wrong += 1
        return nmean_wrong, Zvec

    def countWrongSDs(self, sds, expectedsds, xrange):
        nsd_wrong = 0
        Yvec = np.sqrt(self.nrepeats / 2) * (np.power(sds, 2) / np.power(expectedsds, 2) - 1)
        for n, Y in enumerate(Yvec):
            if (Y < xrange[0] or Y > xrange[1]):
                nsd_wrong += 1
            elif (np.isnan(Y)):
                if not sds[n] == expectedsds[n]:
                    nsd_wrong += 1
            elif (np.isinf(Y)):
                nsd_wrong += 1
        self.assertLessEqual(nsd_wrong, 5)
        return nsd_wrong, Yvec

    def readSettingsFile(self, stochdir, tnum):
        # Read the Settings file
        setfile = stochdir + "/" + tnum + "-settings.txt"
        settings = {}
        with open(setfile) as f:
            for line in f.readlines():
                if ":" in line:
                    line = line.replace(" ", "")
                    lvec = line.strip().split(":")
                    settings[lvec[0]] = lvec[1]
        f.close()
        if "start" not in settings:
            self.fail("Missing 'start' setting for test \"{}\"".format(tnum))
        if "duration" not in settings:
            self.fail("Missing 'duration' setting for test \"{}\"".format(tnum))
        if "steps" not in settings:
            self.fail("Missing 'steps' setting for test \"{}\"".format(tnum))
        if "meanRange" not in settings:
            self.fail("Missing 'meanRange' setting for test \"{}\"".format(tnum))
        if "sdRange" not in settings:
            self.fail("Missing 'sdRange' setting for test \"{}\"".format(tnum))
        if "variables" not in settings:
            self.fail("Missing 'variables' setting for test \"{}\"".format(tnum))
        if "output" not in settings:
            self.fail("Missing 'output' setting for test \"{}\"".format(tnum))
        self.variables = settings["variables"].split(",")
        self.output = settings["output"].split(",")
        self.start = int(settings["start"])
        self.duration = int(settings["duration"])
        self.steps = int(settings["steps"])
        self.meanRange = eval(settings["meanRange"])
        self.sdRange = eval(settings["sdRange"])

    def readMFile(self, stochdir, tnum):
        # Parse the .m file for more information:
        self.synopsis = "";
        self.testType = "StochasticTimeCourse"
        self.package = "none"
        fname = os.path.join(stochdir, tnum + "-model.m")
        with open(fname) as f:
            for line in f.readlines():
                if "synopsis:" in line:
                    self.synopsis = line.strip().split(":")[1].strip()
                if "testType" in line:
                    self.testType = line.strip().split(":")[1].strip()
                if "packagesPresent" in line:
                    self.package = line.strip().split(":")[1].strip()
        f.close()

    def readExpectedResults(self, stochdir, tnum):

        # Read the expected results
        expectfile = stochdir + "/" + tnum + "-results.csv"
        expected_results = {}
        header = []
        with open(expectfile, "r") as csvfile:
            for lvec in csv.reader(csvfile):
                if "time" in lvec:
                    for header in lvec:
                        expected_results[header] = []
                    header = lvec
                    continue
                for n, entry in enumerate(lvec):
                    expected_results[header[n]].append(float(entry))
        csvfile.close()
        return expected_results

    def getFirstAndLastFilesFrom(self, stochfiles):
        first = ""
        last = ""
        for fname in stochfiles:
            if "l1v2" in fname:
                first = fname
            if "l3v2" in fname:
                last = fname
        if first == "":
            for fname in stochfiles:
                if "l2v1" in fname:
                    first = fname
        if first == "":
            for fname in stochfiles:
                if "l3v1" in fname:
                    first = fname
        self.assertFalse(first == "")
        self.assertFalse(last == "")
        if (platform.release()=="5.4.0-1063-azure"):
            return [last]
        return [first, last]

    def writeResults(self, tnum, lv, nmean_wrong, nsd_wrong, nlnmean_wrong, nlnsd_wrong, time_tot, redo):
        self.results.write(tnum)
        self.results.write("\t" + lv)
        self.results.write("\t" + self.synopsis)
        self.results.write("\t" + str(nmean_wrong))
        self.results.write("\t" + str(nsd_wrong))
        self.results.write("\t" + str(nlnmean_wrong))
        self.results.write("\t" + str(nlnsd_wrong))
        self.results.write("\t" + str(time_tot))
        self.results.write("\t" + str(redo))
        self.results.write("\n")

    def simulateFile(self, fname, tnum, expected_results):
        try:
            rr = roadrunner.RoadRunner(self.stochdir + "/" + tnum + "/" + fname)
        except :
            raise ValueError("Unable to load test file " +  fname + "; perhaps an unknown distrib csymbol.")
        rr.timeCourseSelections = ['time'] + self.variables
        all_results = {}
        means = {}
        sds = {}
        for var in rr.timeCourseSelections:
            all_results[var] = []
            means[var] = []
            sds[var] = []
        for repeat in range(self.nrepeats):
            rr.resetToOrigin()
            if self.testType == "StochasticTimeCourse":
                rr.setIntegrator('gillespie')
                sim = rr.simulate(self.start, self.duration, self.steps + 1)
            elif self.testType == "StatisticalDistribution":
                rr.setIntegrator('cvode')
                rr.getIntegrator().setValue("absolute_tolerance", 1e-9)
                sim = rr.simulate(self.start, self.duration, self.steps + 1)
            else:
                self.fail("Unknown stochastic test type " + self.testType + " in test " + tnum + ".")
            for n, col in enumerate(sim.colnames):
                all_results[col].append(sim[:, n])
        for var in self.variables:
            nmean_wrong = None
            nsd_wrong = None
            nlnmean_wrong = None
            nlnsd_wrong = None
            if (var + "-mean") in self.output:
                means[var] = np.mean(all_results[var], axis=0)
                sdvec = []
                if var + "-sd" in expected_results:
                    sdvec = expected_results[var + "-sd"]
                else:
                    sdvec = np.std(all_results[var], axis=0)
                nmean_wrong, Zvec = self.countWrongMeans(means[var], expected_results[var + "-mean"], sdvec,
                                                         self.meanRange)
                print("Number means for", var, "outside expected range:", nmean_wrong)
            if (var + "-sd") in self.output:
                sds[var] = np.std(all_results[var], axis=0)
                nsd_wrong, Yvec = self.countWrongSDs(sds[var], expected_results[var + "-sd"], self.sdRange)
                print("Number standard deviations for", var, "outside expected range:", nsd_wrong)
            lnvar = "ln(" + var + ")"
            if (lnvar + "-mean") in self.output:
                means[lnvar] = np.mean(np.log(all_results[var]), axis=0)
                nlnmean_wrong, Zvec = self.countWrongMeans(means[lnvar], expected_results[lnvar + "-mean"],
                                                           expected_results[lnvar + "-sd"], self.meanRange)
                print("Number ln-means for", var, "outside expected range:", nlnmean_wrong)
            if (lnvar + "-sd") in self.output:
                sds[lnvar] = np.std(np.log(all_results[var]), axis=0)
                nlnsd_wrong, Yvec = self.countWrongSDs(sds[lnvar], expected_results[lnvar + "-sd"], self.sdRange)
                print("Number ln-standard deviations for", var, "outside expected range:", nlnsd_wrong)
        return (nmean_wrong, nsd_wrong, nlnmean_wrong, nlnsd_wrong)

    def runOneTest(self, tnum):
        if platform.release()=="5.4.0-1063-azure" and (int(tnum)>50 or int(tnum) < 30):
            #This runs a handful of the original stochastic test suite, and the first 10 distrib tests.
            print("Skipping test", tnum, "because it takes too long on azure.")
            return
        print("Running test " + tnum)
        testdir = self.stochdir + "/" + tnum + "/"
        testfiles = []
        for __, _, files in walk(testdir):
            testfiles += files

        self.readSettingsFile(testdir, tnum)
        self.readMFile(testdir, tnum)

        expected_results = self.readExpectedResults(testdir, tnum)
        files = self.getFirstAndLastFilesFrom(testfiles)

        for file in files:
            t1 = time.time()
            redo = False
            nmean_wrong, nsd_wrong, nlnmean_wrong, nlnsd_wrong = self.simulateFile(file, tnum, expected_results)
            if (nmean_wrong and nmean_wrong > 3) or \
                    (nsd_wrong and nsd_wrong > 5) or \
                    (nlnmean_wrong and nlnmean_wrong > 3) or \
                    (nlnsd_wrong and nlnsd_wrong > 5):
                # Re-run it to see if we just got unlucky:
                (nmean_wrong, nsd_wrong, nlmean_wrong, nlnds_wrong) = self.simulateFile(file, tnum, expected_results)
                redo = True
            time_tot = time.time() - t1
            self.writeResults(tnum, file, nmean_wrong, nsd_wrong, nlnmean_wrong, nlnsd_wrong, time_tot, redo)
            print("Test file", file, "total time:", time_tot)
            if nmean_wrong is not None:
                self.assertLessEqual(nmean_wrong, 3)
            if nsd_wrong is not None:
                self.assertLessEqual(nsd_wrong, 5)
            if nlnmean_wrong is not None:
                self.assertLessEqual(nlnmean_wrong, 3)
            if nlnsd_wrong is not None:
                self.assertLessEqual(nlnsd_wrong, 5)

    def run_all_stoch_tests(self):
        for stochdir in self.stochdirs:
            stochfiles = []
            for __, _, files in walk(stochdir):
                stochfiles += files

            tnum = stochdir.split('/')[-1]
            if "0" not in tnum:
                continue
            if self.onlysometests and tnum not in self.sometests:
                continue
            print("Running test ", tnum)
            self.runOneTest(tnum)

    # Run this and then paste into source code below.
    def createRunTestSource(self):
        for n in range(100):
            tnum = str(n + 1).zfill(5)
            print("    def test_stoch_" + tnum + "(self):")
            print("        self.runOneTest('" + tnum + "')")
            print("")

    def test_stoch_00001(self):
        self.runOneTest('00001')

    def test_stoch_00002(self):
        self.runOneTest('00002')

    # def test_stoch_00003(self):
    #     self.runOneTest('00003')

    def test_stoch_00004(self):
        self.runOneTest('00004')

    # This test passes, but takes a ridiculously long time to do so.
    # def test_stoch_00005(self):
    #     self.runOneTest('00005')

    def test_stoch_00006(self):
        self.runOneTest('00006')

    def test_stoch_00007(self):
        self.runOneTest('00007')

    def test_stoch_00008(self):
        self.runOneTest('00008')

    def test_stoch_00009(self):
        self.runOneTest('00009')

    def test_stoch_00010(self):
        self.runOneTest('00010')

    def test_stoch_00011(self):
        self.runOneTest('00011')

    def test_stoch_00012(self):
        self.runOneTest('00012')

    def test_stoch_00013(self):
        self.runOneTest('00013')

    def test_stoch_00014(self):
        self.runOneTest('00014')

    def test_stoch_00015(self):
        self.runOneTest('00015')

    def test_stoch_00016(self):
        self.runOneTest('00016')

    def test_stoch_00017(self):
        self.runOneTest('00017')

    def test_stoch_00018(self):
        self.runOneTest('00018')

    def test_stoch_00019(self):
        self.runOneTest('00019')

    def test_stoch_00020(self):
        self.runOneTest('00020')

    def test_stoch_00021(self):
        self.runOneTest('00021')

    def test_stoch_00022(self):
        self.runOneTest('00022')

    # This test passes, but takes a ridiculously long time to do so.
    # def test_stoch_00023(self):
    #     self.runOneTest('00023')

    def test_stoch_00024(self):
        self.runOneTest('00024')

    def test_stoch_00025(self):
        self.runOneTest('00025')

    def test_stoch_00026(self):
        self.runOneTest('00026')

    def test_stoch_00027(self):
        self.runOneTest('00027')

    # def test_stoch_00028(self):
    #     self.runOneTest('00028')

    # def test_stoch_00029(self):
    #     self.runOneTest('00029')

    def test_stoch_00030(self):
        self.runOneTest('00030')

    def test_stoch_00031(self):
        self.runOneTest('00031')

    # def test_stoch_00032(self):
    #     self.runOneTest('00032')

    def test_stoch_00033(self):
        self.runOneTest('00033')

    def test_stoch_00034(self):
        self.runOneTest('00034')

    def test_stoch_00035(self):
        self.runOneTest('00035')

    def test_stoch_00036(self):
        self.runOneTest('00036')

    def test_stoch_00037(self):
        self.runOneTest('00037')

    def test_stoch_00038(self):
        self.runOneTest('00038')

    def test_stoch_00039(self):
        self.runOneTest('00039')

    def test_stoch_00040(self):
        self.runOneTest('00040')

    def test_stoch_00041(self):
        self.runOneTest('00041')

    def test_stoch_00042(self):
        self.runOneTest('00042')

    def test_stoch_00043(self):
        self.runOneTest('00043')

    def test_stoch_00044(self):
        self.runOneTest('00044')

    def test_stoch_00045(self):
        self.runOneTest('00045')

    def test_stoch_00046(self):
        self.runOneTest('00046')

    def test_stoch_00047(self):
        self.runOneTest('00047')

    def test_stoch_00048(self):
        self.runOneTest('00048')

    def test_stoch_00049(self):
        self.runOneTest('00049')

    def test_stoch_00050(self):
        self.runOneTest('00050')

    def test_stoch_00051(self):
        self.runOneTest('00051')

    def test_stoch_00052(self):
        self.runOneTest('00052')

    def test_stoch_00053(self):
        self.runOneTest('00053')

    def test_stoch_00054(self):
        self.runOneTest('00054')

    def test_stoch_00055(self):
        self.runOneTest('00055')

    def test_stoch_00056(self):
        self.runOneTest('00056')

    def test_stoch_00057(self):
        self.runOneTest('00057')

    def test_stoch_00058(self):
        self.runOneTest('00058')

    def test_stoch_00059(self):
        self.runOneTest('00059')

    def test_stoch_00060(self):
        self.runOneTest('00060')

    def test_stoch_00061(self):
        self.runOneTest('00061')

    def test_stoch_00062(self):
        self.runOneTest('00062')

    def test_stoch_00063(self):
        self.runOneTest('00063')

    def test_stoch_00064(self):
        self.runOneTest('00064')

    def test_stoch_00065(self):
        self.runOneTest('00065')

    def test_stoch_00066(self):
        self.runOneTest('00066')

    def test_stoch_00067(self):
        self.runOneTest('00067')

    def test_stoch_00068(self):
        self.runOneTest('00068')

    def test_stoch_00069(self):
        self.runOneTest('00069')

    def test_stoch_00070(self):
        self.runOneTest('00070')

    def test_stoch_00071(self):
        self.runOneTest('00071')

    def test_stoch_00072(self):
        self.runOneTest('00072')

    def test_stoch_00073(self):
        self.runOneTest('00073')

    def test_stoch_00074(self):
        self.runOneTest('00074')

    def test_stoch_00075(self):
        self.runOneTest('00075')

    def test_stoch_00076(self):
        self.runOneTest('00076')

    def test_stoch_00077(self):
        self.runOneTest('00077')

    def test_stoch_00078(self):
        self.runOneTest('00078')

    def test_stoch_00079(self):
        self.runOneTest('00079')

    def test_stoch_00080(self):
        self.runOneTest('00080')

    def test_stoch_00081(self):
        self.runOneTest('00081')

    def test_stoch_00082(self):
        self.runOneTest('00082')

    def test_stoch_00083(self):
        self.runOneTest('00083')

    def test_stoch_00084(self):
        self.runOneTest('00084')

    def test_stoch_00085(self):
        self.runOneTest('00085')

    def test_stoch_00086(self):
        self.runOneTest('00086')

    def test_stoch_00087(self):
        self.runOneTest('00087')

    def test_stoch_00088(self):
        self.runOneTest('00088')

    def test_stoch_00089(self):
        self.runOneTest('00089')

    def test_stoch_00090(self):
        self.runOneTest('00090')

    def test_stoch_00091(self):
        self.runOneTest('00091')

    def test_stoch_00092(self):
        self.runOneTest('00092')

    def test_stoch_00093(self):
        self.runOneTest('00093')

    def test_stoch_00094(self):
        self.runOneTest('00094')

    def test_stoch_00095(self):
        self.runOneTest('00095')

    def test_stoch_00096(self):
        self.runOneTest('00096')

    def test_stoch_00097(self):
        self.runOneTest('00097')

    def test_stoch_00098(self):
        self.runOneTest('00098')

    def test_stoch_00099(self):
        self.runOneTest('00099')

    def test_stoch_00100(self):
        self.runOneTest('00100')


if __name__ == "__main__":
    unittest.main()
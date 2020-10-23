# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:39:24 2020

@author: Lucian Smith
"""

from __future__ import division
from os import walk, scandir

import csv
import numpy as np
import roadrunner

stochdir = "C:/Users/Lucian/Desktop/stochastic-distrib-test-suite/"
outfile = "results.tsv"
nrepeats = 10000

onlysometests = False
sometests = ["00057"]

stochdirs = [f.path for f in scandir(stochdir) if f.is_dir()]


results = open(stochdir + outfile, "w")
results.write("Test #")
results.write("\tLevel/Version")
results.write("\tSynopsis")
results.write("\tnWrongMeans")
results.write("\tnWrongSDs")
results.write("\tnWrongLogMeans")
results.write("\tnWrongLogSDs")
results.write("\n")

def countWrongMeans(means, expectedmeans, expectedsds, xrange):
    nmean_wrong = 0
    Zvec = np.sqrt(nrepeats) * (means - expectedmeans) / expectedsds
    for Z in Zvec:
        if Z<xrange[0] or Z>xrange[1]:
            nmean_wrong += 1 
    return nmean_wrong, Zvec

def countWrongSDs(sds, expectedsds, xrange):
    nsd_wrong = 0
    Yvec = np.sqrt(nrepeats/2) * (np.power(sds,2)/np.power(expectedsds,2) - 1)
    for n, Y in enumerate(Yvec):
        if (Y < xrange[0] or Y>xrange[1]):
            nsd_wrong += 1
        elif (np.isnan(Y)):
            if not sds[n] == expectedsds[n]:
                nsd_wrong += 1
        elif (np.isinf(Y)):
            nsd_wrong += 1
    return nsd_wrong, Yvec

for stochdir in stochdirs:
    stochfiles= []
    for __, _, files in walk(stochdir):
        stochfiles += files
    
    tnum = stochdir.split('/')[-1]
    if "0" not in tnum:
        continue
    if onlysometests and tnum not in sometests:
        continue
    print("Running test", tnum)
    
    #Read the Settings
    setfile = stochdir + "/" + tnum + "-settings.txt"
    settings = {}
    for line in open(setfile, "r"):
        if ":" in line:
            line = line.replace(" ","")
            lvec = line.strip().split(":")
            settings[lvec[0]] = lvec[1]
    if "start" not in settings:
        print("Missing 'start' setting for test", tnum)
        continue
    if "duration" not in settings:
        print("Missing 'duration' setting for test", tnum)
        continue
    if "steps" not in settings:
        print("Missing 'steps' setting for test", tnum)
        continue
    if "meanRange" not in settings:
        print("Missing 'meanRange' setting for test", tnum)
        continue
    if "sdRange" not in settings:
        print("Missing 'sdRange' setting for test", tnum)
        continue
    if "variables" not in settings:
        print("Missing 'variables' setting for test", tnum)
        continue
    if "output" not in settings:
        print("Missing 'output' setting for test", tnum)
        continue
    variables = settings["variables"].split(",")
    output = settings["output"].split(",")
    start = int(settings["start"])
    duration = int(settings["duration"])
    steps = int(settings["steps"])
    meanRange = eval(settings["meanRange"])
    sdRange = eval(settings["sdRange"])
    
    #Get synopsis from .m file
    synopsis = "";
    testType = "StochasticTimeCourse"
    for line in open(stochdir + "/" + tnum + "-model.m", "r"):
        if "synopsis:" in line:
            synopsis = line.strip().split(":")[1].strip()
        if "testType" in line:
            testType = line.strip().split(":")[1].strip()
    
            
    #Read the expected results
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

    #Perform the simulation.
    for fname in stochfiles:
        if "xml" not in fname:
            continue
        lv = "l3v1"
        if "l3v2" in fname:
            lv = "l3v2"
        results.write(tnum)
        results.write("\t" + lv)
        results.write("\t" + synopsis)
        try:
            rr = roadrunner.RoadRunner(stochdir + "/" + fname)
        except:
            print("Unable to simulate test", fname, ": unknown distrib csymbol.  Probably.")
            results.write("\tUnable to simulate model\n")
            continue
        rr.timeCourseSelections = variables
        all_results = {}
        means = {}
        sds = {}
        for var in rr.timeCourseSelections:
            all_results[var] = []
            means[var] = []
            sds[var] = []
        for repeat in range(nrepeats):
            rr.resetToOrigin()
            sim = rr.simulate(start, duration, steps+1)
            for n, col in enumerate(sim.colnames):
                all_results[col].append(sim[:, n])
        for var in rr.timeCourseSelections:
            nmean_wrong = ""
            nsd_wrong = ""
            nlnmean_wrong = ""
            nlnsd_wrong = ""
            if (var + "-mean") in output:
                means[var] = np.mean(all_results[var], axis=0)
                sdvec = []
                if var + "-sd" in expected_results:
                    sdvec = expected_results[var + "-sd"]
                else:
                    sdvec = np.std(all_results[var], axis=0)
                nmean_wrong, Zvec = countWrongMeans(means[var], expected_results[var + "-mean"], sdvec, meanRange)
                print("Number means outside expected range:", nmean_wrong)
            if (var + "-sd") in output:
                sds[var] = np.std(all_results[var], axis=0)
                nsd_wrong, Yvec = countWrongSDs(sds[var], expected_results[var + "-sd"], sdRange)
                print("Number standard deviations outside expected range:", nsd_wrong)
            lnvar = "ln(" + var + ")"
            if (lnvar + "-mean") in output:
                means[lnvar] = np.mean(np.log(all_results[var]), axis=0)
                nlnmean_wrong, Zvec = countWrongMeans(means[lnvar], expected_results[lnvar + "-mean"], expected_results[lnvar + "-sd"], meanRange)
                print("Number ln-means outside expected range:", nlnmean_wrong)
            if (lnvar + "-sd") in output:
                sds[lnvar] = np.std(np.log(all_results[var]), axis=0)
                nlnsd_wrong, Yvec = countWrongSDs(sds[lnvar], expected_results[lnvar + "-sd"], sdRange)
                print("Number ln-standard deviations outside expected range:", nlnsd_wrong)
            
            results.write("\t" + str(nmean_wrong))
            results.write("\t" + str(nsd_wrong))
            results.write("\t" + str(nlnmean_wrong))
            results.write("\t" + str(nlnsd_wrong))
            results.write("\n")
            
results.close()

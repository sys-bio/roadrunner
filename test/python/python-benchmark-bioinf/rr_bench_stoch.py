#!/usr/bin/env python

# Roadrunner benchmarking script, Apr 2015
# Authors: J Kyle Medley
# Language: Python 2.7.9
# Usage: cd <this_dir> && python rr_bench_stoch.py >rr-stoch-results.csv

from __future__ import print_function
from roadrunner import *
import time
import csv
from sys import stderr, stdout

tests = [ \
  ('stoch', "./stoch/stoch_l2v4.xml")]

start = 0;
end = 10;
steps = 10;

csvwriter = csv.writer(stdout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

def timeit(name, path):
  print('Model: {}'.format(name), file=stderr)
  startTime = time.time()
  r=RoadRunner(path)
  loadTime = time.time()
  m=r.simulate(start,end,steps,integrator='gillespie')
  endTime = time.time()

  csvwriter.writerow([name, loadTime-startTime, endTime-loadTime, endTime-startTime])


for n in range(3):
  print('Trial {}'.format(n+1), file=stderr)
  csvwriter.writerow(['Trial {}'.format(n+1)])
  csvwriter.writerow(['Name', 'Load_time', 'Run_time', 'Total_time'])
  for t in tests:
      timeit(*t)


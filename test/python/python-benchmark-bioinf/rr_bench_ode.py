#!/usr/bin/env python

# Roadrunner benchmarking script, Apr 2015
# Authors: Andy Somogyi, J Kyle Medley
# Language: Python 2.7.9
# Usage: cd <this_dir> && python bench.py >rr-ode-results.csv

from __future__ import print_function
import roadrunner
import time
import csv
from sys import stderr, stdout, exit

# Set tolerance parameters (must match across simulators)
absolute_tol_default = 1.000000e-007
rel_tol_default = 0.0001

tests = [ \
  ('jean_marie', "./jean_marie/Jean_Marie_AMPA16_RobHow_v6.xml", 0.25, 0.024),
  ('jana_wolf', "./jana_wolf/Jana_WolfGlycolysis.xml", 0.3, 0.357),
  ('biomod14', "./biomod14/BIOMD0000000014.xml", 1.000, 0.04),
  ('biomod33', "./biomod33/BIOMD0000000033.xml", 0.191, 0.09),
  ('00001',   "./00001/00001-sbml-l2v4.xml",     0.084, 0.03),
  ('00002',   "./00002/00002-sbml-l2v4.xml",     0.086, 0.03),
  ('00050',  "./00050/00050-sbml-l2v4.xml",      0.389, 0.06),
  ('00100', "./00100/00100-sbml-l2v4.xml",       0.829, 0.07),
  ('00150', "./00150/00150-sbml-l2v4.xml",       1.230, 0.07),
  ('00200', "./00200/00200-sbml-l2v4.xml",       1.806, 0.07),
  ('00250', "./00250/00250-sbml-l2v4.xml",       3.236, 0.08),
  ('00300', "./00300/00300-sbml-l2v4.xml",       3.188, 0.08),
  ('00350', "./00350/00350-sbml-l2v4.xml",       4.130, 0.08),
  ('00400', "./00400/00400-sbml-l2v4.xml",       4.735, 0.10),
  ('00450', "./00450/00450-sbml-l2v4.xml",       5.211, 0.10),
  ('00500', "./00500/00500-sbml-l2v4.xml",       6.209, 0.10)]

start = 0;
end = 50;
steps = 50;

csvwriter = csv.writer(stdout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# set to true if performance regression detected
regDetected = False

def formatTime(t):
  return round(t,4)

def timeit(name, path, loadTimeCap, runTimeCap):
  print('Model: {}'.format(name), file=stderr)
  startTime = time.time()
  global regDetected
  passFail = 'pass'

  # Load the model
  r=roadrunner.RoadRunner(path)

  # use these settings for consistent performance
  r.getIntegrator().setValue('relative_tolerance', rel_tol_default)
  r.getIntegrator().setValue('absolute_tolerance', absolute_tol_default)
  r.getIntegrator().setValue('stiff', False)

  loadTime = time.time()
  if loadTime-startTime > loadTimeCap:
    regDetected = True
    passFail = '***FAIL***'

  m=r.simulate(start, end, steps)
  endTime = time.time()
  if endTime-loadTime > runTimeCap:
    regDetected = True
    passFail = '***FAIL***'

  csvwriter.writerow([name, formatTime(loadTime-startTime), formatTime(endTime-loadTime), formatTime(endTime-startTime), passFail])

# First row in output: Version info
print(roadrunner.__version__, file=stderr)
csvwriter.writerow([roadrunner.__version__])

for n in range(1):
  print('Trial {}'.format(n+1), file=stderr)
  csvwriter.writerow(['Trial {}'.format(n+1)])
  csvwriter.writerow(['Name', 'Load_time', 'Run_time', 'Total_time'])
  for t in tests:
      timeit(*t)

if regDetected == True:
  exit(1)
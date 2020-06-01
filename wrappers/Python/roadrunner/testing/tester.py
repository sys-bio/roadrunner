# Test Module for RoadRunner
#
# Usage:
# import rrTester
# runTester (pathtoModelFile, modelFileName)


#-------------------------------------------------------------
# Tests for steady state and stoichiometric calculations in
# roadRunner. Herbert M Sauro November 2012
# Nov 2013: Modified to test Andy's SWIG API
#-------------------------------------------------------------

#------------------------------------------
# Change this line for different test files
#nameOfResultsFile = 'results_roadRunnerTest_1.txt'

from __future__ import print_function

import random
import string
import roadrunner
from roadrunner import Config
from roadrunner import Logger
import re
import numpy
from numpy import *
import os

# Module wide file handle
fHandle = ''
rpadding = 45
#gFailedTests = 0
#gPassedTests = 0

# --------------------------------------------------------------------------
# SUPPORT ROUTINES
# --------------------------------------------------------------------------

def expectApproximately (a, b, tol):
    diff = a - b
    return abs(diff) < tol

def expectArrayApproximately(a, b, tol):
    diff = a - b
    return (abs(diff) < tol).all()

def passMsg (errorFlag, msg = ""):
    global gFailedTests
    global gPassedTests
    if msg:
        msg = "\n    " + msg
    if not errorFlag:
        gPassedTests = gPassedTests+1
        return "PASS"
    gFailedTests = gFailedTests+1
    return "*****FAIL*****" + msg

# Empty lines are ignored
# Lines starting with # are also ignored

def readLine ():
    line = fHandle.readline()

    while (line == '') or (line[0] == '#') or (line[0] == '\n'):
        if line == '':
            return line
        line = fHandle.readline();
    return line.strip('\n')

def divide(line):
    words = line.strip()
    words = words.strip('"')
    words = re.compile("[ \t=]").split(words)
    while "" in words: words.remove("")
    return words

def jumpToNextTest():
    line = readLine()
    #line = ''
    #while line == '':
    #      line = fHandle.readline().strip ('\n')
    while (line != "") and (line[0] != '['):
        line = readLine()
    # remove any leading or trailing whitespace
    return line.strip()

def getSBMLStr ():
    sbmlStr = ''
    line = fHandle.readline()
    while (line[0] != '['):
        sbmlStr = sbmlStr + line
        line = fHandle.readline()
    return sbmlStr, line.strip()

# ------------------------------------------------------------------------
# TESTS START HERE
# ------------------------------------------------------------------------

def checkMatrixVsUpcomingText(calc):
    errorFlag = False
    msg = ""
    rows = len(calc)
    if (not(isinstance(calc[0], numpy.ndarray))):
        compareUpcomingValuesWith(calc, 1e-4)
        return
    for i in range(0,rows):
        line = readLine ()
        words = line.split()
        cols = len(calc[0])
        if(len(words) != cols):
            msg = "Expected a row of length " + str(len(words)) + ", but calculated a row of length " + str(cols)
            errorFlag = True
            break
        for j in range(0,cols):
            expectedValue = float(words[j])
            calcValue = calc[i,j]
            if expectApproximately (expectedValue, calcValue, abs(1e-7+calcValue)*1E-4) == False:
                msg = "Expected " + words[j] + ", but calculated " + str(calcValue)
                errorFlag = True
                break
    print(passMsg (errorFlag, msg))


def checkComplexVecVsUpcomingText(calc):
    errorFlag = False
    msg = ""
    rows = len(calc)
    for i in range(0,rows):
        line = readLine ()
        words = line.split()
        if(len(words) != 2):
            msg = "Faulty test:  needed a row of length two for complex number, but found " + str(len(words))
            errorFlag = True
            break
            expectedValue =  complex(float(words[0], float(words[1])))
            if expectApproximately (expectedValue, calc[i], abs(calc+1e-7)*1E-4) == False:
                msg = "Expected " + words[j] + ", but calculated " + str(calc[i,j])
                errorFlag = True
                break
    print(passMsg (errorFlag, msg))


def compareUpcomingValuesWith(ss, error):
    errorFlag = False
    msg = ""
    words = divide(readLine())
    if (len(words)) == 1:
        for i in range(len(ss)):
            if expectApproximately(float (words[0]), ss[i], abs(ss[i]+1e-5)*error) == False:
                msg = "Expected " + words[0] + ", but calculated " + str(ss[i])
                errorFlag = True
                break;
            if i < len(ss) - 1:
                words = divide(readLine())
    else:
        for i in range (len (ss)):
            if expectApproximately(float (words[i]), ss[i], abs(ss[i]+1e-5)*error) == False:
                msg = "Expected " + words[i] + ", but calculated " + str(ss[i])
                errorFlag = True
                break;
    print(passMsg (errorFlag, msg))

def setConservationLaw(rrInstance, testId):
    line = readLine ()
    if line == 'True':
        rrInstance.conservedMoietyAnalysis = True
    else:
        rrInstance.conservedMoietyAnalysis = False

def checkSpeciesConcentrations(rrInstance, testId):
    words = []
    species = []
    m = rrInstance.model.getNumFloatingSpecies()
    for i in range (0,m):
        words = divide(readLine())
        words.append (rrInstance.model[words[0]])
        species.append (words)

    # Steady State Concentrations
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    for i in range (0,m):
        expectedValue =  float (species[i][1])
        if expectApproximately (expectedValue, species[i][2], abs(species[i][2]*1E-4)) == False:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkSteadyStateFluxes(rrInstance, testId):
    words = []
    fluxes = []
    # Steady State Fluxes
    rrInstance.reset()
    print("Computing Steady State.  Distance to SteadyState:", rrInstance.steadyState())
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.model.getNumReactions();
    for i in range (0,n):
        line = readLine ()
        words = line.split("=")
        words.append (rrInstance.model[words[0]])
        fluxes.append (words)

    for i in range (0,n):
        expectedValue = float (fluxes[i][1])
        if expectApproximately (expectedValue, fluxes[i][2], abs((expectedValue+(1e-3))*1E-5)) == False:
            errorFlag = True
            break
    print(passMsg (errorFlag))

def checkFullJacobian(rrInstance, testId):
    # Jacobian
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    Jacobian = rrInstance.getFullJacobian()
    checkMatrixVsUpcomingText(Jacobian)


def checkReducedJacobian(rrInstance, testId):
    # Jacobian
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    Jacobian = rrInstance.getReducedJacobian()
    checkMatrixVsUpcomingText(Jacobian)


def checkAmountJacobian(rrInstance, testId):
    # Jacobian
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    Jacobian = rrInstance.getFullJacobian()
    checkMatrixVsUpcomingText(Jacobian)


def checkIndividualEigenvalues(rrInstance, testId):
    # Eigenvalues
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    m = len(rrInstance.getEigenValueIds())//3
    try:
        for i in range(0,int(m)):
            words = divide(readLine())
            eigenvalueName = words[0]
            calc = rrInstance.getValue ('eigenReal(' + eigenvalueName + ')')
            calc = complex(calc, rrInstance.getValue('eigenImag(' + eigenvalueName + ')'))
            expected = complex(float(words[1]), float(words[2]))

            if expectApproximately (calc, expected, abs(calc+1e-7)*1E-4) == False:
                errorFlag = True
                break
        print(passMsg (errorFlag))
    except (Exception) as e:
        print('Unexpected error in checkIndividualEigenvalues:' + str(e))


def checkIndividualAmountEigenvalues(rrInstance, testId):
    # Eigenvalues
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    m = len(rrInstance.getEigenValueIds())//3
    try:
        for i in range(0,int(m)):
            words = divide(readLine())
            eigenvalueName = words[0]
            realPart = rrInstance.getValue ('eigenReal(' + eigenvalueName + ')')
            realPart = float (realPart)

            if expectApproximately (realPart, float(words[1]), 1E-5) == False:
                errorFlag = True
                break
        print(passMsg (errorFlag))
    except (Exception) as e:
        print('Unexpected error in checkIndividualEigenvalues:' + str(e))


def checkEigenvalueMatrix(rrInstance, testId):
    # Eigenvalues
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    eigenvalues = rrInstance.getFullEigenValues()
    checkComplexVecVsUpcomingText(eigenvalues)


def checkReducedEigenvalueMatrix(rrInstance, testId):
    # Eigenvalues
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    eigenvalues = rrInstance.getReducedEigenValues()
    checkComplexVecVsUpcomingText(eigenvalues)


def checkEigenvalueAmountMatrix(rrInstance, testId):
    # Eigenvalues
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    eigenvalues = rrInstance.getFullEigenValues()
    checkComplexVecVsUpcomingText(eigenvalues)

def checkStoichiometryMatrix(rrInstance, testId):
    # Stoichiometry matrix
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getFullStoichiometryMatrix()
    checkMatrixVsUpcomingText(st)

def checkReducedStoichiometryMatrix(rrInstance, testId):
    # Stoichiometry matrix
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getReducedStoichiometryMatrix()
    checkMatrixVsUpcomingText(st)

def checkLinkMatrix(rrInstance, testId):
    # Link matrix
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getLinkMatrix()
    checkMatrixVsUpcomingText(st)

def checkUnscaledConcentrationControlMatrix(rrInstance, testId):
    # Unscaled Concentration Control matrix
    rrInstance.resetToOrigin()
    Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getUnscaledConcentrationControlCoefficientMatrix();
    checkMatrixVsUpcomingText(st)


def checkScaledConcentrationControlMatrix(rrInstance, testId):
    # Unscaled Concentration Control matrix
    rrInstance.resetToOrigin()
    Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getScaledConcentrationControlCoefficientMatrix();
    checkMatrixVsUpcomingText(st)


def checkUnscaledFluxControlCoefficientMatrix(rrInstance, testId):
    # Unscaled Flux Control matrix
    rrInstance.resetToOrigin()
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getUnscaledFluxControlCoefficientMatrix();
    checkMatrixVsUpcomingText(st)


def checkScaledFluxControlCoefficientMatrix(rrInstance, testId):
    # Scaled Flux Control matrix
    rrInstance.resetToOrigin()
    print(("Check " + testId).ljust( rpadding), end="")
    st = rrInstance.getScaledFluxControlCoefficientMatrix()
    checkMatrixVsUpcomingText(st)


def checkUnscaledElasticityMatrix(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    uee = rrInstance.getUnscaledElasticityMatrix()
    checkMatrixVsUpcomingText(uee)

def checkUnscaledElasticityAmountMatrix(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    uee = rrInstance.getUnscaledElasticityMatrix()
    checkMatrixVsUpcomingText(uee)

def checkScaledElasticityMatrix(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
    ee = rrInstance.getScaledElasticityMatrix()
    checkMatrixVsUpcomingText(ee)


def checkScaledElasticityAmountMatrix(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    roadrunner.Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    ee = rrInstance.getScaledElasticityMatrix()
    checkMatrixVsUpcomingText(ee)


def checkGetFloatingSpeciesIds(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    line = readLine ()
    words = line.split()
    expected = rrInstance.model.getFloatingSpeciesIds()
    m = rrInstance.model.getNumFloatingSpecies()
    for i in range(0,m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))

def checkGetBoundarySpeciesIds(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    line = readLine ()
    words = line.split()
    expected = rrInstance.model.getBoundarySpeciesIds()
    m = rrInstance.model.getNumBoundarySpecies()
    for i in range(0,m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkGetGlobalParameterIds (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    line = readLine ()
    words = line.split()
    expected = rrInstance.model.getGlobalParameterIds()
    m = rrInstance.model.getNumGlobalParameters()
    for i in range(0,m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkGetCompartmentIds (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    line = readLine ()
    words = line.split()
    expected = rrInstance.model.getCompartmentIds()
    m = rrInstance.model.getNumCompartments()
    for i in range(0,m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkReactionIds (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    expected = rrInstance.model.getReactionIds()
    m = rrInstance.model.getNumReactions();
    for i in range(0,m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkFloatingSpeciesInitialConcentrationIds (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")

    words = divide(readLine())

    expected = rrInstance.model.getFloatingSpeciesInitConcentrationIds()


    print(passMsg (words != expected))


def checkEigenValueIds (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    expected = rrInstance.getEigenValueIds()
    m = rrInstance.model.getNumFloatingSpecies()
    for i in range(0,3*m):
        if words[i] != expected[i]:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkSetSteadyStateSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    rrInstance.conservedMoietyAnalysis = True
    rrInstance.steadyStateSelections = words
    print(passMsg (errorFlag))
    time = rrInstance.steadyState()
    if (time>0.00001):
        time = rrInstance.steadyState()
    print("Computing Steady State.  Distance to SteadyState:", time)


def checkGetSteadyStateSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    words = divide(readLine())
    result = str(rrInstance.steadyStateSelections)

    print(passMsg (result == words))


def checkSetTimeCourseSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    result = rrInstance.selections = words
    if result == False:
        errorFlag = True
    print(passMsg (errorFlag))


def checkGetTimeCourseSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    words = divide(readLine())
    result = str(rrInstance.selections)

    print(passMsg (result == words))

def checkComputeSteadyStateValues(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.getSteadyStateValues()
    compareUpcomingValuesWith(ss, 1E-6)


def checkFloatingSpeciesConcentrations(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.model.getFloatingSpeciesConcentrations()
    compareUpcomingValuesWith(ss, 1E-6)


def checkBoundarySpeciesConcentrations(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.model.getBoundarySpeciesConcentrations()
    compareUpcomingValuesWith(ss, 1E-6)


def checkGlobalParameterValues(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.model.getGlobalParameterValues()
    compareUpcomingValuesWith(ss, 1E-6)


def checkInitalFloatingSpeciesConcentations(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.model.getFloatingSpeciesInitConcentrations()

    words = readLine().split()

    same = len(words) == len(ss) and \
        len(words) == sum([1 for i,j in zip(words,ss) if expectApproximately(float (i), j, 1E-6)])

    print(passMsg (not same))


def checkReactionRates(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    ss = rrInstance.model.getReactionRates()
    compareUpcomingValuesWith(ss, 1E-4)


def checkGetReactionRatesByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = readLine().split()
    n = rrInstance.model.getNumReactions()
    for i in range (n):
        value = rrInstance.model.getReactionRates()[i]
        if expectApproximately(float (words[i]), value, 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))

def checkGetSpeciesInitialConcentrationsByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    n = rrInstance.model.getNumFloatingSpecies()
    values = rrInstance.model.getFloatingSpeciesInitConcentrations()
    for i in range (n):
        if expectApproximately(float (words[i]), values[i], 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))


def checkSetSpeciesInitialConcentrationsByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    n = rrInstance.model.getNumFloatingSpecies()
    rrInstance.model.setFloatingSpeciesInitConcentrations(array(list(map(float, words))))
    rt = rrInstance.model.getFloatingSpeciesInitConcentrations()
    for i in range (n):
        if expectApproximately(float (words[i]), rt[i], 1E-6) == False:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkSetSpeciesInitialConcentrations(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.model.getNumFloatingSpecies()
    for i in range (0,n):
        words = divide(readLine())
        if rrInstance.setValue(words[0], float(words[1])) == False:
            errorFlag = True
            break
        if expectApproximately(rrInstance.getValue(words[0]), float (words[1]), 1E-6) == False:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkNumberOfDependentSpecies(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = int (readLine())
    n = rrInstance.model.getNumDepFloatingSpecies()
    if n != value:
        errorFlag = True
    print(passMsg (errorFlag))


def checkNumberOfIndependentSpecies(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = int (readLine())
    n = rrInstance.model.getNumIndFloatingSpecies()
    if n != value:
        errorFlag = True
    print(passMsg (errorFlag))


def checkNumberOfRateRules(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = int (readLine())
    if rrInstance.model.getNumRateRules() != value:
        errorFlag = True
    print(passMsg (errorFlag))


def checkGetRatesOfChange(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    values = rrInstance.getRatesOfChange()
    for i in range (len(words)):
        if expectApproximately (float (words[i]), values[i], abs(values[i]+1e-7)*1E-5) == False:
            errorFlag = True
    print(passMsg (errorFlag))


def checkGetReactionRatesEx (rrInstance, testId):
    # Pass on Python
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    print(passMsg (errorFlag))


def checkGetRatesOfChangeEx (rrInstance, testId):
    # Pass on Python
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    print(passMsg (errorFlag))


def checkGetRatesOfChangeByIndex(rrInstance, testId):
    # Pass on Python
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    print(passMsg (errorFlag))


def checkGetRatesOfChangeIds(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    line = readLine ()
    words = line.split()
    expected = rrInstance.model.getFloatingSpeciesIds()
    m = rrInstance.model.getNumFloatingSpecies()
    for i in range(0,m):
        if words[i] != expected[i] + "'":
            errorFlag = True
            break
    print(passMsg (errorFlag))


def checkRK4Output(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    rrInstance.reset(roadrunner.SelectionRecord.ALL)
    cvode = rrInstance.simulate(0, 10)
    rrInstance.reset()
    rrInstance.setIntegrator('rk4')
    rk4 = rrInstance.simulate(0, 10)
    rrInstance.setIntegrator('cvode')
    if not expectArrayApproximately(cvode[-1], rk4[-1], 1E-6):
        errorFlag = True
    print(passMsg (errorFlag))
    
    
def checkRK45Output(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    rrInstance.reset(roadrunner.SelectionRecord.ALL)
    cvode = rrInstance.simulate(0, 10)
    rrInstance.reset()
    rrInstance.setIntegrator('rk45')
    rk45 = rrInstance.simulate(0, 10)
    rrInstance.setIntegrator('cvode')
    if not expectArrayApproximately(cvode[-1], rk45[-1], 1E-6):
        errorFlag = True
    print(passMsg (errorFlag))
    
    
# ---------------------------------------------------------------------------

def setGetValues(rrInstance, IdList, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    for i in range (len(IdList)):
        value = random.random()*10
        rrInstance.model[dList[i]] = value
        if expectApproximately (rrInstance.model[IdList[i]], value, 1E-6) == False:
            errorFlag = True
            break
    print(passMsg (errorFlag))


def setGetTimeStart(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = random.random ()*10
    rrInstance.setTimeStart (value)
    if expectApproximately (rrInstance.getTimeStart (), value, 1E-6) == False:
        errorFlag = True
    print(passMsg (errorFlag))


def setGetTimeEnd(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = random.random ()*10
    rrInstance.setTimeEnd (value)
    if expectApproximately (rrInstance.getTimeEnd (), value, 1E-6) == False:
        errorFlag = True
    print(passMsg (errorFlag))


def setGetNumberOfPoints(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    value = random.randint (1, 100)
    rrInstance.setNumPoints (value)
    if rrInstance.getNumPoints () != value:
        errorFlag = True
    print(passMsg (errorFlag))


def setGetTimeCourseSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    myList = rrInstance.getFloatingSpeciesIds()
    newList = list (myList)
    random.shuffle (newList)
    rrInstance.setTimeCourseSelectionList (newList)
    if rrInstance.getTimeCourseSelectionList() != newList:
        errorFlag = True
    print(passMsg (errorFlag))


def setGetSteadyStateSelectionList(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    myList = rrInstance.getFloatingSpeciesIds()
    newList = list (myList)
    while newList == myList:
        random.shuffle (newList)
    rrInstance.setSteadyStateSelectionList (newList)
    getList = rrInstance.getSteadyStateSelectionList()
    if getList != newList:
        errorFlag = True
    print(passMsg (errorFlag))


def setGetFloatingSpeciesByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.getNumFloatingSpecies()
    for i in range (n):
        value = random.random()*10
        rrInstance.setFloatingSpeciesByIndex (i, value)
        if expectApproximately(rrInstance.getFloatingSpeciesByIndex (i), value, 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))


def setGetBoundarySpeciesByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.getNumBoundarySpecies()
    for i in range (n):
        value = random.random()*10
        rrInstance.setBoundarySpeciesByIndex (i, value)
        if expectApproximately(rrInstance.getBoundarySpeciesByIndex (i), value, 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))


def setGetCompartmentByIndex(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.getNumCompartments()
    for i in range (n):
        value = random.random()*10
        rrInstance.setCompartmentByIndex (i, value)
        if expectApproximately(rrInstance.getCompartmentByIndex (i), value, 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))


def setGetGlobalParameterByIndex (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    n = rrInstance.getNumberOfGlobalParameters()
    for i in range (n):
        value = random.random()*10
        rrInstance.setGlobalParameterByIndex (i, value)
        if expectApproximately(rrInstance.getGlobalParameterByIndex (i), value, 1E-6) == False:
            errorFlag = True
            break;
    print(passMsg (errorFlag))


def setGetFloatingSpeciesConcentrations (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    getArray = rrInstance.getFloatingSpeciesConcentrations()
    setArray = zeros(len(getArray))
    for i in range(len(getArray)):
        value = random.random()*10
        setArray[i] = value
    rrInstance.setFloatingSpeciesConcentrations (setArray)
    if (setArray != rrInstance.getFloatingSpeciesConcentrations()).all():
        errorFlag = True
    print(passMsg (errorFlag))


def setGetBoundarySpeciesConcentrations (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    getArray = rrInstance.getBoundarySpeciesConcentrations()
    setArray = zeros(len(getArray))
    for i in range(len(getArray)):
        value = random.random()*10
        setArray[i] = value
    rrInstance.setBoundarySpeciesConcentrations (rrInstance.PythonArrayTorrVector (setArray))
    if (setArray != rrInstance.getBoundarySpeciesConcentrations()).all():
        errorFlag = True
    print(passMsg (errorFlag))


def setGetInitialFloatingSpeciesConcentrations (rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    getArray = rrInstance.getFloatingSpeciesInitialConcentrations ()
    setArray = zeros(len(getArray))
    for i in range(len(getArray)):
        value = random.random()*10
        setArray[i] = value
    rrInstance.setFloatingSpeciesInitialConcentrations (setArray)
    if (setArray != rrInstance.getFloatingSpeciesInitialConcentrations()).all():
        errorFlag = True
    print(passMsg (errorFlag))


def setGetReset(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    values = zeros (rrInstance.getNumberOfFloatingSpecies())
    for i in range (len (values)):
        values[i] = random.random()*10
    initial = rrInstance.getFloatingSpeciesInitialConcentrations()
    rrInstance.setFloatingSpeciesConcentrations (values)
    # Should reset the floats by to the current initial condition
    rrInstance.reset()
    values = rrInstance.getFloatingSpeciesConcentrations()
    if(values != initial).all():
        errorFlag = True
    print(passMsg (errorFlag))

def testReset(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    rrInstance.setValue(words[0], float(words[1]))
    rrInstance.setValue(words[2], float(words[3]))
    rrInstance.setValue(words[4], float(words[5]))
    rrInstance.reset()
    f = rrInstance.getValue(words[0])
    k = rrInstance.getValue(words[2])
    d = rrInstance.getValue(words[4])
    if (f == float(words[1])) or (k != float(words[3])) or (d != float(words[5])):
        errorFlag = True
    print(passMsg (errorFlag))
    
def testResetAll(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    rrInstance.setValue(words[0], float(words[1]))
    rrInstance.setValue(words[2], float(words[3]))
    rrInstance.resetAll()
    k = rrInstance.getValue(words[0])
    d = rrInstance.getValue(words[2])
    if (k == float(words[1])) or (d != float(words[3])):
        errorFlag = True
    print(passMsg (errorFlag))
    
def testResetToOrigin(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    rrInstance.setValue(words[0], float(words[1]))
    rrInstance.resetToOrigin()
    d = rrInstance.getValue(words[0])
    rrInstance.reset()
    if (d == float(words[1])):
        errorFlag = True
    print(passMsg (errorFlag))
    
def testResetConservedTotal(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    ct = words[0]
    rrInstance.conservedMoietyAnalysis = True
    rrInstance.steadyState()
    rrInstance.setValue(ct, 1000)
    rrInstance.reset()
    ct_val = rrInstance.getValue(ct)
    if ct_val == 1000:
        errorFlag = True
    print(passMsg (errorFlag))

def testSetValues(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    spe = words[:int(len(words)/2)]
    val = words[int(len(words)/2):]
    val = [float(i) for i in val]
    n = rrInstance.setValues(spe, val)
    for i in range(len(spe)):
        if not (rrInstance.getValue(spe[i]) == val[i]):
            errorFlag = True
    rrInstance.reset()
    print(passMsg (errorFlag))

def checkJacobian(rrInstance, testId):
    # TODO need to update python 2.x printing
    print(("Check " + testId).ljust( rpadding), end="")
    import numpy as n

    errors = False

    # max difference between reduced and full
    maxDiff = 2e-10

    # save the old value
    saved = Config.getValue(Config.ROADRUNNER_JACOBIAN_MODE)

    # set to amounts mode
    Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    full = rrInstance.getFullJacobian()
    reduced = rrInstance.getReducedJacobian()

    m = n.max(n.abs(reduced-full))
    if (m > maxDiff):
        errors = True
        print("Jacobians different in amounts mode, max difference: " + str(m))

    # set to conc mode
    Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)

    full = rrInstance.getFullJacobian()
    reduced = rrInstance.getReducedJacobian()

    m = n.max(n.abs(reduced-full))
    if (m > maxDiff):
        errors = True
        print("Jacobians different in concentrations mode, max difference: " + str(m))

    # restore previous value
    Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, saved)

    print(passMsg(errors))

def checkControlCoefficient(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    n = rrInstance.getCC(words[0], words[1])
    if abs(n - float(words[2])) > 1e-6:
        errorFlag = True
    print(passMsg (errorFlag))

def checkVariableEndTime(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = True
    words = divide(readLine())

    try:
        # passing variableStep to simulate should throw
        rrInstance.simulate(float(words[0]), float(words[1]), variableStep=True)
    except (TypeError):
        errorFlag = False

    rrInstance.getIntegrator().resetSettings()
    rrInstance.getIntegrator().setValue('variable_step_size', True)
    n2 = rrInstance.simulate(float(words[0]), float(words[1]))

    if expectApproximately(n2[-1][0], float(words[1]), 1e-16) == False:
        errorFlag = True

    print(passMsg (errorFlag))

def checkDefaultTimeStep(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    words = divide(readLine())
    n = rrInstance.simulate()
    if n.shape[0] != int(words[0]):
        print('expected {} rows but found {}'.format(int(words[0]), n.shape[0]))
        errorFlag = True
    print(passMsg (errorFlag))

def checkSimulateTimepointsVsIntervals(rrInstance, testId):
    '''
    Third positional argument is number of points.
    Steps keyword argument is number of intervals.
    '''
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    try:
        n1 = rrInstance.simulate(0, 10, steps=1)
        if n1.shape[0] != 2:
            errorFlag = True
        n2 = rrInstance.simulate(0, 10, 2)
        if n2.shape[0] != 2:
            errorFlag = True
    except:
        errorFlag = True
    try:
        m = rrInstance.simulate(0, 100, 51)
        n = rrInstance.simulate(0, 100, points=51)
        if n.shape[0] != m.shape[0]:
            errorFlag = True
        n = rrInstance.simulate(start=0, end=100, points=51)
        if n.shape[0] != m.shape[0]:
            errorFlag = True
        n = rrInstance.simulate(0, 100, steps=50)
        if n.shape[0] != m.shape[0]:
            errorFlag = True
        n = rrInstance.simulate(start=0, end=100, steps=50)
        if n.shape[0] != m.shape[0]:
            errorFlag = True

        if len(rrInstance.model.getFloatingSpeciesIds()) < 1:
            errorFlag = True
        else:
            spec_id = rrInstance.model.getFloatingSpeciesIds()[0]
            m = rrInstance.simulate(0, 100, 51, ['time', spec_id])
            n = rrInstance.simulate(0, 100, points=51, selections=['time', spec_id])
            if n.shape[0] != m.shape[0]:
                errorFlag = True
            n = rrInstance.simulate(start=0, end=100, points=51, selections=['time', spec_id])
            if n.shape[0] != m.shape[0]:
                errorFlag = True
            n = rrInstance.simulate(0, 100, steps=50, selections=['time', spec_id])
            if n.shape[0] != m.shape[0]:
                errorFlag = True
            n = rrInstance.simulate(start=0, end=100, steps=50, selections=['time', spec_id])
            if n.shape[0] != m.shape[0]:
                errorFlag = True
    except:
        errorFlag = True
    print(passMsg (errorFlag))

def checkMonotonicTimepoints(rrInstance, testId):
    '''
    Timepoint values should only increase.
    No two timepoints should ever have the same time value.
    '''
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    try:
        words = divide(readLine())
        startTime = float(words[0])
        endTime = float(words[1])
        npoints = int(words[2])

        # For variable step
        rrInstance.getIntegrator().setValue('variable_step_size', True)
        n = rrInstance.simulate(startTime, endTime)
        for k in range(1,n.shape[0]):
            if n[k-1,0] >= n[k,0]:
                print('Monotonicity failure var step: {} >= {}'.format(n[k-1,0], n[k,0]))
                errorFlag = True

        # For fixed step
        rrInstance.reset()
        rrInstance.getIntegrator().setValue('variable_step_size', False)
        n = rrInstance.simulate(startTime, endTime, npoints)
        for k in range(1,n.shape[0]):
            if n[k-1,0] >= n[k,0]:
                print('Monotonicity failure fixed step: {} >= {}'.format(n[k-1,0], n[k,0]))
                errorFlag = True
    except (Exception) as e:
        print('Caught exception: {}'.format(e))
        errorFlag = True
    print(passMsg (errorFlag))

def checkEventPreandPostfireTimepoints(rrInstance, testId):
    '''
    Timepoint values should only increase.
    No two timepoints should ever have the same time value.
    '''
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    try:
        words = divide(readLine())

        def find_close_timepoint(simdata, t, tol):
            '''
            Finds the index of the closest timepoint in the
            simulation results which is within tol of t
            '''
            for k in range(simdata.shape[0]):
                if abs(simdata[k,0] - t) <= tol:
                    return k
            return -1;

        rrInstance.getIntegrator().setValue('variable_step_size', True)
        simresult = rrInstance.simulate(0, 10)

        max_tol = 0.0001

        # iterate through each expected event timepoint
        for w in words:
            t = float(w)
            # expect two timepoints close by
            i = find_close_timepoint(simresult, t, max_tol)
            if i < 0 or i+1 >= simresult.shape[0]:
                errorFlag = True
            else:
                if abs(simresult[i+1,0] - simresult[i,0]) > max_tol:
                    errorFlag = True
    except (Exception) as e:
        print('Caught exception: {}'.format(e))
        errorFlag = True
    print(passMsg (errorFlag))

def checkGillespieSeed(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    rrInstance.setIntegrator('gillespie')
    words = divide(readLine())
    rrInstance.getIntegrator().setValue('seed', words[0])
    arr1 = rrInstance.simulate(0,100,steps=10)
    rrInstance.reset()
    rrInstance.getIntegrator().setValue('seed', words[1])
    arr2 = rrInstance.simulate(0,100,steps=10)
    if arr1[1,0] == arr2[1,0]:
        errorFlag = True
    rrInstance.setIntegrator('cvode')
    print(passMsg (errorFlag))

def checkGillespieValue(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    rrInstance.setIntegrator('gillespie')
    n = rrInstance.simulate(0,100,steps=10)
    if n[-1,1] != n[-2,1]:
        errorFlag = True
    rrInstance.setIntegrator('cvode')
    print(passMsg (errorFlag))
    
def checkGillespieFixedSteps(rrInstance, testId):
    print(("Check " + testId).ljust( rpadding), end="")
    errorFlag = False
    rrInstance.setIntegrator('gillespie')
    rrInstance.setIntegratorSetting('gillespie', 'variable_step_size', False)
    n = rrInstance.simulate(0,100,steps=10)
    if any(n[:,0] != numpy.linspace(0, 100, 11)):
        errorFlag = True
    rrInstance.setIntegrator('cvode')
    print(passMsg (errorFlag))

def unitTestIntegratorSettings(testDir):
    errorFlag = False

    r = roadrunner.RoadRunner(os.path.join(testDir,'Test_1.xml'))

    r.setIntegrator('cvode')
    r.getIntegratorByName('gillespie').seed = 12345
    r.setIntegrator('gillespie')
    if r.getIntegrator().getSetting('seed') != 12345:
        errorFlag = True

    r.setIntegrator('cvode')
    r.setIntegratorSetting('gillespie', 'seed', 54321)
    r.setIntegrator('gillespie')
    if r.getIntegrator().getSetting('seed') != 54321:
        errorFlag = True

    print(passMsg (errorFlag))

def addSpecies(rrInstance):
    words = []
    words = divide(readLine())
    print("Add species " + words[0])
    if (len(words)==2):
        rrInstance.addSpecies(words[0], words[1])
    elif (len(words)==3):
        rrInstance.addSpecies(words[0], words[1], float(words[2]))
    elif (len(words)==4):
        rrInstance.addSpecies(words[0], words[1], float(words[2]), bool(words[3]))
    elif (len(words)==5):
        rrInstance.addSpecies(words[0], words[1], float(words[2]), bool(words[3]), bool(words[4]))
    elif (len(words)==6):
        rrInstance.addSpecies(words[0], words[1], float(words[2]), bool(words[3]), bool(words[4]), words[5])
    elif (len(words)==7):
        rrInstance.addSpecies(words[0], words[1], float(words[2]), bool(words[3]), bool(words[4]), words[5], bool(words[6]))
    else:
        print(passMsg(False, "Incorrect number of arguments given to addSpecies."))


def addReaction(rrInstance):
    words = []
    words = divide(readLine())
    print("Add reaction " + words[0])
    reactants = words[1].split(',')
    products = words[2].split(',')
    rrInstance.addReaction(words[0], reactants, products, words[3], bool(words[4]))


def addParameter(rrInstance):
    words = []
    words = divide(readLine())
    print("Add parameter " + words[0])
    rrInstance.addParameter(words[0], float(words[1]), bool(words[2]))

def addCompartment(rrInstance):
    words = []
    words = divide(readLine())
    print("Add compartment " + words[0])
    rrInstance.addCompartment(words[0], float(words[1]), bool(words[2]))

def addAssignmentRule(rrInstance):
    words = []
    words = divide(readLine())
    print("Add assignment rule " + words[0])
    rrInstance.addAssignmentRule(words[0], words[1], bool(words[2]))

def addRateRule(rrInstance):
    words = []
    words = divide(readLine())
    print("Add rate rule " + words[0])
    rrInstance.addRateRule(words[0], words[1], bool(words[2]))

def addEvent(rrInstance):
    words = []
    words = divide(readLine())
    print("Add event " + words[0])
    rrInstance.addEvent(words[0], bool(words[1]), words[2], bool(words[3]))

def addTrigger(rrInstance):
    words = []
    words = divide(readLine())
    print("Add trigger for " + words[0])
    rrInstance.addTrigger(words[0], words[1], bool(words[2]))


def addPriority(rrInstance):
    words = []
    words = divide(readLine())
    print("Add priority for " + words[0])
    rrInstance.addPriority(words[0], words[1], bool(words[2]))

def addDelay(rrInstance):
    words = []
    words = divide(readLine())
    print("Add delay for " + words[0])
    rrInstance.addDelay(words[0], words[1], bool(words[2]))

def addEventAssignment(rrInstance):
    words = []
    words = divide(readLine())
    print("Add event assignment for " + words[1])
    rrInstance.addEventAssignment(words[0], words[1], words[2], bool(words[3]))


def removeSpecies(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove species " + words[0])
    rrInstance.removeSpecies(words[0], bool(words[1]))

def removeReaction(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove reaction " + words[0])
    rrInstance.removeReaction(words[0], bool(words[1]))

def removeParameter(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove parameter " + words[0])
    rrInstance.removeParameter(words[0], bool(words[1]))

def removeCompartment(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove compartment " + words[0])
    rrInstance.removeCompartment(words[0], bool(words[1]))

def removeRules(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove rules for " + words[0])
    rrInstance.removeRules(words[0], bool(words[1]), bool(words[2]))

def removeEvent(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove event " + words[0])
    rrInstance.removeEvent(words[0], bool(words[1]))

def removeEventAssignments(rrInstance):
    words = []
    words = divide(readLine())
    print("Remove event assignments for" + words[1])
    rrInstance.removeEventAssignments(words[0], words[1], bool(words[2]))

def setBoundary(rrInstance):
    words = []
    words = divide(readLine())
    print("Set boundary condition for" + words[0])
    rrInstance.setBoundary(words[0], bool(words[1]), bool(words[2]))

def setHasOnlySubstanceUnits(rrInstance):
    words = []
    words = divide(readLine())
    print("Set HasOnlySubstanceUnits for" + words[0])
    rrInstance.setHasOnlySubstanceUnits(words[0], bool(words[1]), bool(words[2]))

def setInitAmount(rrInstance):
    words = []
    words = divide(readLine())
    print("Set initial amount for" + words[0])
    rrInstance.setInitAmount(words[0], words[1], bool(words[2]))

def setInitConcentration(rrInstance):
    words = []
    words = divide(readLine())
    print("Set initial concentration for" + words[0])
    rrInstance.setInitConcentration(words[0], words[1], bool(words[2]))

def setConstant(rrInstance):
    words = []
    words = divide(readLine())
    print("Set constant attribute for" + words[0])
    rrInstance.setConstant(words[0], bool(words[1]), bool(words[2]))

def setReversible(rrInstance):
    words = []
    words = divide(readLine())
    print("Set reversible attribute for" + words[0])
    rrInstance.setReversible(words[0], bool(words[1]), bool(words[2]))

def setPersistent(rrInstance):
    words = []
    words = divide(readLine())
    print("Set persistent attribute for" + words[0])
    rrInstance.setPersistent(words[0], bool(words[1]), bool(words[2]))

def setTriggerInitialValue(rrInstance):
    words = []
    words = divide(readLine())
    print("Set TriggerInitialValue attribute for" + words[0])
    rrInstance.setPersistent(words[0], bool(words[1]), bool(words[2]))

def setKineticLaw(rrInstance):
    words = []
    words = divide(readLine())
    print("Set kinetic law for" + words[0])
    rrInstance.setKineticLaw(words[0], words[1], bool(words[2]))

def setVectorAmountAbsoluteTolerance(rrInstance, testId):
    errorFlag = False
    value = divide(readLine())
    floatValue = [float(i) for i in value]
    rrInstance.getIntegrator().setValue('absolute_tolerance', floatValue)
    tolerance = rrInstance.getIntegrator().getValue("absolute_tolerance")
    expected = divide(readLine())
    if not (len(tolerance) == len(expected)):
        errorFlag = True
    for i in range(len(expected)):
        if not (tolerance[i] == float(expected[i])):
            errorFlag = True
    print(passMsg (errorFlag))


def setScalarConcentrationAbsoluteTolerance(rrInstance, testId):
    errorFlag = False
    value = float(readLine())
    rrInstance.getIntegrator().setConcentrationTolerance(value)
    tolerance = rrInstance.getIntegrator().getValue("absolute_tolerance")
    expected = divide(readLine())
    if not (len(tolerance) == len(expected)):
        errorFlag = True
    for i in range(len(expected)):
        if not (tolerance[i] == float(expected[i])):
            errorFlag = True
    print(passMsg (errorFlag))

def setVectorConcentrationAbsoluteTolerance(rrInstance, testId):
    errorFlag = False
    value = divide(readLine())
    floatValue = [float(i) for i in value]
    rrInstance.getIntegrator().setConcentrationTolerance(floatValue)
    tolerance = rrInstance.getIntegrator().getValue("absolute_tolerance")
    expected = divide(readLine())
    if not (len(tolerance) == len(expected)):
        errorFlag = True
    for i in range(len(expected)):
        if not (tolerance[i] == float(expected[i])):
            errorFlag = True
    print(passMsg (errorFlag))


def setIndividualTolerance(rrInstance, testId):
    errorFlag = False
    value = float(readLine())
    rrInstance.getIntegrator().setIndividualTolerance(value)
    tolerance = rrInstance.getIntegrator().getValue("absolute_tolerance")
    expected = divide(readLine())
    if not (len(tolerance) == len(expected)):
        errorFlag = True
    for i in range(len(expected)):
        if not (tolerance[i] == float(expected[i])):
            errorFlag = True
    print(passMsg (errorFlag))



def scriptTests():
    print("\nTesting Set and Get Functions")
    print("-----------------------------")
    setGetValues(rrInstance.getFloatingSpeciesIds(), 'Set/Get Value (Floats)')
    setGetValues(rrInstance.getBoundarySpeciesIds(), 'Set/Get Value (Boundary)')
    setGetValues(rrInstance.getGlobalParameterIds(), 'Set/Get Value (Global Parameters)')
    setGetValues(rrInstance.getCompartmentIds(), 'Set/Get Value (Compartments)')
    setGetTimeStart('Set/Get TimeStart')
    setGetTimeEnd ('Set/Get TimeEnd')
    setGetNumberOfPoints ('Set/Get Number Of Points')
    setGetTimeCourseSelectionList ('Set/Get Time Course Selection List')
    setGetSteadyStateSelectionList ('Set/Get Steady State Selection List')
    setGetFloatingSpeciesByIndex ('Set/Get Floating Species by Index')
    setGetBoundarySpeciesByIndex ('Set/Get Boundary Species by Index')
    setGetCompartmentByIndex ('Set/Get Compartment by Index')
    setGetGlobalParameterByIndex ('Set/Get Global Parameter buy Index')
    setGetBoundarySpeciesConcentrations ('Set/Get Boundary Species Concs')
    setGetFloatingSpeciesConcentrations ('Set/Get Floating Species Concs')
    setGetInitialFloatingSpeciesConcentrations ('Set/Get Initial Concs')
    setGetReset ('Set/Get Reset Method')


# ------------------------------------------------------------------------
# List of tests
functions = {'[Add Species]' : addSpecies,
             '[Add Reaction]' : addReaction,
             '[Add Parameter]' : addParameter,
             '[Add Compartment]' : addCompartment,
             '[Add Assignment Rule]' : addAssignmentRule,
             '[Add Rate Rule]' : addRateRule,
             '[Add Event]' : addEvent,
             '[Add Trigger]' : addTrigger,
             '[Add Priority]' : addPriority,
             '[Add Delay]' : addDelay,
             '[Add Event Assignment]' : addEventAssignment,
             '[Amount/Concentration Jacobians]' : checkJacobian,
             '[Amount Jacobian]' : checkAmountJacobian,
             '[Boundary Species Concentrations]': checkBoundarySpeciesConcentrations,
             '[Boundary Species Ids]': checkGetBoundarySpeciesIds,
             '[Check Default Time Step]': checkDefaultTimeStep,
             '[Check Monotonic Timepoints]': checkMonotonicTimepoints,
             '[Check Event Pre and Postfire Timepoints]': checkEventPreandPostfireTimepoints,
             '[Check RK4 Output]': checkRK4Output,
             '[Check RK45 Output]': checkRK45Output,
             '[Check Simulate Points vs Steps]': checkSimulateTimepointsVsIntervals,
             '[Compartment Ids]': checkGetCompartmentIds,
             '[Compute Steady State Values]': checkComputeSteadyStateValues,
             '[Conservation Laws]': setConservationLaw,
             '[Eigenvalue Matrix]': checkEigenvalueMatrix,
             '[Eigenvalue Amount Matrix]': checkEigenvalueAmountMatrix,
             '[Floating Species Concentrations]': checkFloatingSpeciesConcentrations,
             '[Floating Species Ids]': checkGetFloatingSpeciesIds,
             '[Full Jacobian]': checkFullJacobian,
             '[Get Control Coefficient]': checkControlCoefficient,
             '[Get Eigenvalue Ids]': checkEigenValueIds,
             '[Get Variable End Time]': checkVariableEndTime,
             '[Get Global Parameter Values]': checkGlobalParameterValues,
             '[Get Initial Floating Species Concs]': checkInitalFloatingSpeciesConcentations,
             '[Get Rates of Change By Index]': checkGetRatesOfChangeByIndex,
             '[Get Rates Of Change]': checkGetRatesOfChange,
             '[Get Rates Of Change Ids]': checkGetRatesOfChangeIds,
             '[Get Rates of Change Ex]': checkGetRatesOfChangeEx,
             '[Get Reaction Rates By Index]': checkGetReactionRatesByIndex,
             '[Get Reaction Rates Ex]': checkGetReactionRatesEx,
             '[Get Reaction Rates]': checkReactionRates,
             '[Get Species Initial Concentrations By Index]': checkGetSpeciesInitialConcentrationsByIndex,
             '[Get Steady State Selection List]': checkGetSteadyStateSelectionList,
             '[Get Steady State Selection List 2]': checkGetSteadyStateSelectionList,
             '[Get Time Course Selection List]': checkGetTimeCourseSelectionList,
             '[Gillespie Seed]': checkGillespieSeed,
             '[Gillespie Value]': checkGillespieValue,
             '[Gillespie Fixed Time Interval]': checkGillespieFixedSteps,
             '[Global Parameter Ids]': checkGetGlobalParameterIds,
             '[Individual Eigenvalues]': checkIndividualEigenvalues,
             '[Individual Amount Eigenvalues]': checkIndividualAmountEigenvalues,
             '[Link Matrix]': checkLinkMatrix,
             '[Number of Dependent Species]': checkNumberOfDependentSpecies,
             '[Number of Independent Species]': checkNumberOfIndependentSpecies,
             '[Number Of Rate Rules]': checkNumberOfRateRules,
             '[Reaction Ids]': checkReactionIds,
             '[Reduced Eigenvalue Matrix]': checkReducedEigenvalueMatrix,
             '[Reduced Jacobian]': checkReducedJacobian,
             '[Reduced Stoichiometry Matrix]': checkReducedStoichiometryMatrix,
             '[Remove Species]': removeSpecies,
             '[Remove Reaction]': removeReaction,
             '[Remove Parameter]': removeParameter,
             '[Remove Compartment]': removeCompartment,
             '[Remove Rules]': removeRules,
             '[Remove Event]': removeEvent,
             '[Remove Event Assignments]': removeEventAssignments,
             '[Scaled Concentration Control Matrix]': checkScaledConcentrationControlMatrix,
             '[Scaled Elasticity Matrix]': checkScaledElasticityMatrix,
             '[Scaled Elasticity Amount Matrix]': checkScaledElasticityAmountMatrix,
             '[Scaled Flux Control Matrix]': checkScaledFluxControlCoefficientMatrix,
             '[Set Boundary]': setBoundary,
             '[Set HasOnlySubstanceUnits]': setHasOnlySubstanceUnits,
             '[Set Initial Amount]': setInitAmount,
             '[Set Initial Concentration]': setInitConcentration,
             '[Set Constant]': setConstant,
             '[Set Reversible]': setReversible, 
             '[Set Kinetic Law]': setKineticLaw,
             '[Set Persistent]': setPersistent,
             '[Set Trigger Initial Value]': setTriggerInitialValue,
             '[Set Species Initial Concentrations By Index]': checkSetSpeciesInitialConcentrationsByIndex,
             '[Set Species Initial Concentrations]': checkSetSpeciesInitialConcentrations,
             '[Set Steady State Selection List]': checkSetSteadyStateSelectionList,
             '[Set Steady State Selection List 2]': checkSetSteadyStateSelectionList,
             '[Set Time Course Selection List]': checkSetTimeCourseSelectionList,
             '[Set Vector Amount Absolute Tolerance]': setVectorAmountAbsoluteTolerance,
             '[Set Scalar Concentration Absolute Tolerance]': setScalarConcentrationAbsoluteTolerance,
             '[Set Vector Concentration Absolute Tolerance]': setVectorConcentrationAbsoluteTolerance,
             '[Set Individual Tolerance]': setIndividualTolerance,
             '[Species Concentrations]': checkSpeciesConcentrations,
             '[Species Initial Concentration Ids]': checkFloatingSpeciesInitialConcentrationIds,
             '[Steady State Fluxes]': checkSteadyStateFluxes,
             '[Stoichiometry Matrix]': checkStoichiometryMatrix,
             '[Test Reset]': testReset,
             '[Test ResetAll]': testResetAll,
             '[Test ResetToOrigin]': testResetToOrigin,
             '[Test ResetConservedTotal]': testResetConservedTotal,
             '[Test SetValues]': testSetValues,
             '[Unscaled Concentration Control Matrix]': checkUnscaledConcentrationControlMatrix,
             '[Unscaled Elasticity Matrix]': checkUnscaledElasticityMatrix,
             '[Unscaled Elasticity Amount Matrix]': checkUnscaledElasticityAmountMatrix,
             '[Unscaled Flux Control Matrix]': checkUnscaledFluxControlCoefficientMatrix,
              }


def getDefaultTestDir():
    import os.path as p
    d = p.dirname(__file__)
    return p.join(d,"test_data")


def runTester (testDir=None):
    """
    Run a series of tests from a testing dir.

    This enumerates all the files in a directory, and all the files ending with '.rrtest'
    are assumed to be testing files.
    """
    global fHandle
    global gFailedTests
    global gPassedTests

    import os.path as p
    from glob import glob

    gFailedTests = 0
    gPassedTests = 0

    if testDir is None:
        testDir = getDefaultTestDir()

    if not p.isdir(testDir):
        raise Exception("{} is NOT a directory".format(testDir))

    files = glob(p.join(testDir, "*.rrtest"))

    unknownTests = 0
    for file in files:
        print("\n\nStarting Test on ", file)

        # set the globals, these should be class instance vars...
        fHandle = open (file, 'r')
        testId = jumpToNextTest()



        if testId == '[SBML]':
            sbmlStr, testId = getSBMLStr ()
        else:
            Logger.log(Logger.LOG_WARNING, "rrtest file, \""
                       + file + "\" missing SBML section, ignoring test file")
            continue


        # Load any initialization actions
        if testId == '[INITIALIZATION]':
            testId = jumpToNextTest ()
            while testId != '[END_INITIALIZATION]':
                if testId in functions:
                    func = functions[testId]
                    func (rrInstance, testId)
                else:
                    print('No initialization function found for ' + testId)
                testId = jumpToNextTest()
            testId = jumpToNextTest()

        # create a RoadRunner obj with the sbml from the test file
        rrInstance = roadrunner.RoadRunner(sbmlStr)
        print('Successfully loaded model.\n')

        # Model editing functions
        if testId == '[EDITING]':
            testId = jumpToNextTest ()
            while testId != '[END_EDITING]':
                if testId in functions:
                    func = functions[testId]
                    func (rrInstance)
                else:
                    print('No initialization function found for ' + testId)
                testId = jumpToNextTest()
            testId = jumpToNextTest()


        # Now start the tests proper
        while testId != '':
            if testId in functions:
                func = functions[testId]
                func(rrInstance, testId)
            else:
                #getFloatingSpeciesAmountRates
                unknownTests = unknownTests+1
                print((testId).ljust( rpadding), 'UNKNOWN TEST')
            testId = jumpToNextTest()

    for testFunc in [unitTestIntegratorSettings]:
      testFunc(testDir)

    print("\n\nTotal failed tests:\t", gFailedTests, \
    "\nTotal unknown tests:\t", unknownTests, \
    "\nTotal passed tests:\t", gPassedTests)

    return gFailedTests

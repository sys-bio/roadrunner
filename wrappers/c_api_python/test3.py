import numpy as np
from ctypes import *

rrLib = cdll.LoadLibrary('../lib/libroadrunner_c_api.so')

print('Set logging level')
rrLib.setLogLevel.restype = c_bool
rrLib.setLogLevel('LOG_DEBUG'.encode('ascii'))

# create the roadrunner instance
rrLib.createRRInstance.restype = c_void_p
rr = rrLib.createRRInstance()

# load the model
rrLib.loadSBMLFromFile.restype = c_bool
result = rrLib.loadSBMLFromFile(rr, '/home/jkm/devel/models/decayModel.xml'.encode('ascii'))
print('  result = {}'.format(result))

# simulate the model
print('Simulate')
rrLib.simulate.restype = c_void_p
rrLib.simulate(rr)

# get the simulation results
rrLib.getSimulationResult.restype = c_void_p
rrLib.getRRDataNumRows.restype = c_int
rrLib.getRRDataNumCols.restype = c_int
rrLib.getMatrixElement.restype = c_bool

##\brief Retrieves an element at a given row and column from a result type variable
#
#Example: status = rrPython.getRRDataElement(result, 2, 4);
#
#\param result A result rrLib
#\param row The row index to the roadrunner  data
#\param column The column index to the roadrunner data
#\return Returns true if succesful
def getRRDataElement(result, row, column):
    value = c_double()
    rvalue = c_int(row)
    cvalue = c_int(column)
    if rrLib.getMatrixElement(result, rvalue, cvalue, pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

def getSimulationResult(aHandle = None):
    if aHandle is None:
        aHandle = gHandle

    result = rrLib.getSimulationResult(aHandle)

    #TODO: Check result
    rowCount = rrLib.getRRDataNumRows(result)
    colCount = rrLib.getRRDataNumCols(result)
    resultArray = np.zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
            resultArray[m, n] = getRRDataElement(result, m, n)
    # DANGER: Missing function freeRRData
    #rrLib.freeRRData(result)
    return resultArray

result = getSimulationResult(rr)
print('Simulation results:')
print(result)
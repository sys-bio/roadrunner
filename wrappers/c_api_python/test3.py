from ctypes import *

rrLib = cdll.LoadLibrary('../lib/libroadrunner_c_api.so')

print('Set logging level')
rrLib.setLogLevel.restype = c_bool
rrLib.setLogLevel('LOG_DEBUG'.encode('ascii'))

rrLib.createRRInstance.restype = c_void_p
rr = rrLib.createRRInstance()

rrLib.loadSBMLFromFile.restype = c_bool
result = rrLib.loadSBMLFromFile(rr, '/home/jkm/devel/models/decayModel.xml'.encode('ascii'))
print('  result = {}'.format(result))

print('Simulate')
rrLib.simulate.restype = c_void_p
rrLib.simulate(rr)
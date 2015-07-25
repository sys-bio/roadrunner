import rrPython

rrPython.setLogLevel('LOG_DEBUG')
print('create instance')
rr = rrPython.createRRInstance()
print('load SBML')
result = rrPython.loadSBMLFromFile('/home/jkm/devel/models/decayModel.xml', rr)
print('  result = {}'.format(result))
print('simulate')
rrPython.simulate(rr)
print('get results')
results = rrPython.getSimulationResult(rr)

print(results)
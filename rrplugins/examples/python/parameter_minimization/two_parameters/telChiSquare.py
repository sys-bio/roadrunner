import roadrunner
import rrplugins
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#Read some 'experimental' data   
expData = rrplugins.DataSeries()

expData = expData.readDataSeries('ExperimentalData.dat')   
           
# Create a roadrunner instance and create some MODEL data
rr = roadrunner.RoadRunner()
rr.load('two_parameters.xml')
rr.setValue('k1', 1.3)
rr.setValue('k2', 2.5)
data = rr.simulate(0, 1.5, 14) 

#Calculate Chi squares
chiPlugin =  rrplugins.Plugin("tel_chisquare")
chiPlugin.NrOfModelParameters = 1
chiPlugin.ExperimentalData    = expData
chiPlugin.ModelData           = rrplugins.getDataSeries(data)
    
if not chiPlugin.execute():
    raise Exception( rrplugins.getLastError() )

print('Chi Square = ', chiPlugin.ChiSquare)
print('Reduced Chi Square = ', chiPlugin.ReducedChiSquare)


#Plot data sets
exNP = chiPlugin.ExperimentalData.toNumPy()
x1 = exNP[:,[0]];    y1 = exNP[:,[1]];   y2 = exNP[:,[2]]
plt.plot(x1,y1, 'o', x1, y2, 'o')

#Simulated data
model = chiPlugin.ModelData.toNumPy()
x1 = model[:,[0]];    y1 = model[:,[1]];   y2 = model[:,[2]]
plt.plot(x1,y1, '-b', x1, y2, '-g')

Chi = round(chiPlugin.ChiSquare,4)
redChi = round(chiPlugin.ReducedChiSquare,4)
  
# plt.text(1.0,0.12, r"\begin{eqnarray*}\chi^2 &=" + str(Chi) + r"\\ \chi^2_{reduced} &=  " + str(redChi) + r"\end{eqnarray*}", fontsize=21, color='red')
plt.show()

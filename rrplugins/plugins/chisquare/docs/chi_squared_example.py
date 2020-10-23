from rrplugins import * 

modelPlugin     = Plugin("tel_test_model") 
noisePlugin     = Plugin("tel_add_noise") 
chiSquarePlugin = Plugin("tel_chisquare") 
 
#Generate internal test data 
modelPlugin.execute() 
modelData = modelPlugin.TestData 
expData = modelPlugin.TestDataWithNoise 
 
chiSquarePlugin.ExperimentalData = expData 
chiSquarePlugin.ModelData = modelData 
chiSquarePlugin.NrOfModelParameters = 1 
 
chiSquarePlugin.execute() 
 
chi = chiSquarePlugin.ChiSquare 
reduced_chi = chiSquarePlugin.ReducedChiSquare 
 
print("ChiSquare is: +", chi)
print("Reduced ChiSquare is: ",reduced_chi)
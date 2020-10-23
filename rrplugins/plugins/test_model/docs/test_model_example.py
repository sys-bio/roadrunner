import rrplugins as tel

modelPlugin = tel.Plugin("tel_test_model") 
 
#Generate internal test data 
modelPlugin.execute() 

test_data = modelPlugin.TestData 
test_data_with_noise = modelPlugin.TestDataWithNoise 
 
test_data.plot() 
test_data_with_noise.plot() 
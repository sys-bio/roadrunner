import roadrunner
import teplugins as tel

try:
    #Retrieve test model from plugin        
    modelPlugin= tel.Plugin("tel_test_model")        
    test_model = modelPlugin.Model
    
    # Create a roadrunner instance and create some data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)    
    data = rr.simulate(0, 10, 511)

    roadrunner.plot(data)           
    print "done"

except Exception as e:
    print 'Problem: ' + `e`
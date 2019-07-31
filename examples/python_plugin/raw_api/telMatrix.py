#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      Totte
#
# Created:     07/03/2014
# Copyright:   (c) Totte 2014
# Licence:     <Exclusive>
#-------------------------------------------------------------------------------

import teplugins as tel

#Get a lmfit plugin object
chiPlugin   = tel.Plugin("tel_chisquare")
lm          = tel.Plugin("tel_levenberg_marquardt")


try:    
    modelPlugin= tel.Plugin("tel_test_model")        
    test_model = modelPlugin.Model
    
    #Setup lmfit properties.
    lm.SBML = test_model
    experimentalData = tel.DataSeries.readDataSeries ("testData.dat")
    lm.ExperimentalData = experimentalData
    
    # Add the parameters that we're going to fit and a initial 'start' value
    lm.setProperty("InputParameterList", ["k1", .3])
    lm.setProperty("FittedDataSelectionList", "[S1] [S2]")
    lm.setProperty("ExperimentalDataSelectionList", "[S1] [S2]")
    
    # Start minimization
    lm.execute()
    
    hessian = lm.getProperty("Hessian")        
    print hessian

except Exception as e:
    print 'Problem.. ' + `e`
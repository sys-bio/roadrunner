***********************
Differential Evolution
***********************

The Differential Evolution plugin is used to fit an SBML model’s parameters to experimental data. The plugin has numerous properties to allow the user full control over the internal fitting engine, as well as access to generated fitted data after a minimization session. In addition, various statistical properties, such as standardized residuals, Q-Q data, ChiSquare and reduced ChiSquare are made accessible to the user. The resulting parameter values also come with estimated confidence limits.
The current implementation is based on the Differential Evolution C++ implementation by Kenneth Price and Rainer Storn.


Start using rrplugins package by importing the library and creating a Plugin instance. 

     >>> from rrplugins import *
     >>> chiPlugin = Plugin("tel_chisquare") 
     >>> modelPlugin = Plugin("tel_test_model") 
     >>> addNoisePlugin = Plugin("tel_add_noise") 
     >>> de = Plugin("differential_evolution")



     >>>try: 
        #========== EVENT FUNCTION SETUP =========================== 
        	def myEvent(dummy): #We are not capturing any data from the plugin
            	print("Iteration, Norm = "+de.getProperty("NrOfIter") + ","+de.getProperty("Norm"))
     >>>#Setup progress event function 
     >>>    progressEvent =  NotifyEventEx(myEvent) 
     >>>    assignOnProgressEvent(de.plugin, progressEvent) 
     >>>    #============================================================  
     >>>    #Create model data, with and without noise using the test_model plugin 
     >>>    modelPlugin.execute() 
   
     >>>    #Setup plugin properties. 
     >>>    de.SBML             = modelPlugin.Model 
     >>>    de.ExperimentalData = modelPlugin.TestDataWithNoise 
      
     >>>    # Add the parameters that we’re going to fit and an initial ’start’ value 
     >>>    de.setProperty("InputParameterList", ["k1", .3]) 
     >>>    de.setProperty("FittedDataSelectionList", "[S1] [S2]") 
     >>>    de.setProperty("ExperimentalDataSelectionList", "[S1] [S2]") 
      
     >>>    # Start minimization 
     >>>    de.execute() 
      
     >>>    print("Minimization finished. \n==== Result ====")
      
     >>>    print("Hessian Matrix")
     >>>    print(de.getProperty("Hessian"))
      
     >>>    print("Covariance  Matrix")
     >>>    print(de.getProperty("CovarianceMatrix"))
      
     >>>    print("ChiSquare = " + de.getProperty("ChiSquare"))
     >>>    print("Reduced ChiSquare = " + de.getProperty("ReducedChiSquare"))
      
     >>>    #This is a list of parameters 
     >>>    parameters = tpc.getPluginProperty (de.plugin, "OutputParameterList") 
     >>>    confLimits = tpc.getPluginProperty (de.plugin, "ConfidenceLimits") 
      
     >>>    #Iterate trough list of parameters and confidence limits 
     >>>    para  = getFirstProperty(parameters) 
     >>>    limit = getFirstProperty(confLimits) 
     >>>    while para and limit: 
     >>>        print (getPropertyName(para) + "=" + getPropertyValue(para) + " +/- " + 			getPropertyValue(limit))
     >>>        para  = getNextProperty(parameters) 
     >>>        limit = getNextProperty(confLimits) 

     >>>    # Get the fitted and residual data 
     >>>    fittedData = de.getProperty ("FittedData").toNumpy 
     >>>    residuals  = de.getProperty ("Residuals").toNumpy 
      
     >>>    # Get the experimental data as a numpy array 
     >>>    experimentalData = modelPlugin.TestDataWithNoise.toNumpy 
      
     >>>    telplugins.plot(fittedData [:,[0,1]], "blue", "-",    "",    "S1 Fitted") 
     >>>    telplugins.plot(fittedData  [:,[0,2]], "blue", "-",    "",    "S2 Fitted") 
     >>>    telplugins.plot(residuals   [:,[0,1]], "blue", "None", "x",   "S1 Residual") 
     >>>    telplugins.plot(residuals   [:,[0,2]], "red",  "None", "x",   "S2 Residual") 
     >>>    telplugins.plot(experimentalData [:,[0,1]], "red",  "",     "*",   "S1 Data") 
     >>>    telplugins.plot(experimentalData [:,[0,2]], "blue", "",     "*",   "S2 Data") 
     >>>    telplugins.plt.show() 
      
     >>>    #Finally, view the manual and version 
     >>>    de.viewManual() 
     >>>    print("Plugin version: " + de.getVersion()) 
      
     >>>except Exception as e: 
        	print("Problem.. " + e) 



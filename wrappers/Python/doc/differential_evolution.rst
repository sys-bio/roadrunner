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



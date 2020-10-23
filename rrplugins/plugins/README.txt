This directory contains Tellurium plugins. 

List of Plugins expected in an official Tellurium release:

* tel_add_noise.dll 
This plugin can be used to generate Gaussian noise and adding it to Tellurium data. This is useful for generating synthetic data sets to test
various algorithms, e.g. parameter fitting algorithms.

* tel_test_model.dll 
This plugin contain an embedded test model. Upon execute(), plugin properties are populated holding test data and test data with noise.

* tel_chisquare.dll  
This plugin is used to calculate the chisquare for two data sets.

* tel_levenberg_marquardt.dll 
This plugin implements the Levenberg-Marquardt optimizer for fitting multi parameter data to models.

* tel_nelder_mead.dll 
This plugin implements the Nelder-Mead optimizer for fitting multi parameter data to models.

* tel_monte_carlo.dll 
This plugin implements a Monte Carlo bootstrap parameter estimation algorithm. 
It has been tested with the levenberg_marquardt and the nelder_mead plugin.

* tel_auto2000.dll 
This plugin serve as a wrapper for the AUTO2000 continuation and bifurcation library. It currently
supports bifurcation problems using one continuation parameter.

See http://tellurium.analogmachine.org/plugins for more details.




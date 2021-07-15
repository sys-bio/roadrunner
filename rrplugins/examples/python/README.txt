Folder containing example/demo scripts demonstrating plugin usage.

* telAddNoise.py                    - Demonstrate usage of the AddNoise plugin
* telTestModel.py                   - Demonstrate usage of the TestModel plugin
* telChiSquare.py                   - Demonstrate usage of the ChiSquare plugin
* telLevenbergMarquardt.py          - Demonstrate usage of the Levenberg-Marquardt plugin
* telNelderMead.py                  - Demonstrate usage of the Nelder-Mead plugin
* telMonteCarlo.py                  - Demonstrate usage of the Monte Carlo bootstrap plugin
* telAuto2000.py                    - Demonstrate usage of the Auto2000 plugin (need bistable.xml on input, also in this folder)

Info: Each of the above examples will attempt to open the plugins embedded PDF documentation on exit. 


The following SUB FOLDERS contain more examples scripts, organized into categories. 
== basic ==
    Contain scripts demonstrating basic interaction with the plugin system

== bifurcation ==
    Contain scripts demonstrating usage of the atuo2000 plugin

== parameter_minimization ==
    Contain scripts demonstrating solving various problems using minimization plugins

    
Any questions regarding the examples may be addressed to Totte Karlsson (totte@dunescientific.com)

==============================================================================================
OBSERVE: Make sure you have properly installed Tellurium plugins in your python environment 
before running any of these scripts. A good test is to execute 'import telplugins', with no quotes, at your python 
prompt. If no message is returned it may serve as an indication of a proper installation.
==============================================================================================

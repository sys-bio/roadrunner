************************
Chi Square Plugin
************************

Introduction
============

The purpose of the ChiSquare plugin is to calculate the ChiSquare and the reduced ChiSquare for two sets of data. 

Plugin Properties
=================

1. ExperimentalData: Data representing Experimental data. 
2. ModelData: Data representing Model data. 
3. NrOfModelParameters: Number of model parameters used to create the model data. 
4. ChiSquare: The calculated ChiSquare. 
5. ReducedChiSquare: The calculated reduced ChiSquare.

Plugin Events
=============

This plugin does not use any plugin events. 

The execute Function
====================

The execute() function will attempt to calculate the ChiSquare, and the reduced ChiSquare, using data supplied by the user. 

Python examples
===============

The python script below shows how to use the plugin. 

.. code-block:: python
   
   from rrplugins import *

   try:   
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
       
       print('ChiSquare is: ' + `chi`)
       print('Reduced ChiSquare is: ' + `reduced_chi`)
       
   except Exception as e:
       print('Problem: ' + `e`)
       

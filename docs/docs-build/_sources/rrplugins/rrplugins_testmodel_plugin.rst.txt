************************
Test Model  Plugin
************************

Introduction
============

The purpose of the TestModel plugin is to conveniently embed a SBML test model in a plugin. In addition, the plugin provides the user with simulated data, with and without applied artificial Gaussian noise.

Currently no settings are exposed for the actual simulation of the test model.

The TestModel plugin depends on the AddNoise plugin. 

Plugin Properties
=================

1. Model: The actual test model, in XML format. 
2. TestData: Simulated data, using the TestModel as input and default RoadRunner Simulation values. 
3. TestDataWithNoise: NumberSimulated data, with applied noise. 

Plugin Events
=============

This plugin does not use any plugin events. 

The execute Function
====================

The execute() function will generate simulated data, and simulated data with noise. The data will be available in the properties, TestData and TestDataWithNoise respectively. 

Python examples
===============

The python script below shows how to use the plugin. 

.. code-block:: python
   
    import rrplugins as tel

    try:    
        modelPlugin = tel.Plugin("tel_test_model")            
        
        #Test model plugin depends on the add_noise plugin
        noisePlugin = tel.Plugin("tel_add_noise")
        
        #Generate internal test data
        modelPlugin.execute()
        test_data = modelPlugin.TestData
        test_data_with_noise = modelPlugin.TestDataWithNoise
            
        test_data.plot()
        test_data_with_noise.plot()
            
    except Exception as e:
        print('Problem: ' + `e`)
        
        
    

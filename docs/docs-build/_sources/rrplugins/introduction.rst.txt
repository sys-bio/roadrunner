*********
rrPlugins
*********

.. toctree::
    :maxdepth: 1
    :hidden:

    rrplugins_architecture
    rrplugins_documentation
    rrplugins_add_noise_plugin
    rrplugins_auto_plugin
    rrplugins_chisquare_plugin
    rrplugins_LM_plugin
    rrplugins_monte_carlo_plugin
    rrplugins_NM_plugin
    rrplugins_testmodel_plugin
    

Introduction
============

The plugin API allows users to easily access roadrunner plugins. Only three concepts need to be understood:

1. Loading a plugin
2. Setting and Getting plugin properties
3. Executing the plugin

Before using the plugin system the plugin library must be imported using the line: 

.. code-block:: python

   import rrplugins as tel

To load a plugin called "add_noise" use the Python command: 

.. code-block:: python

   p = tel.Plugin("tel_add_noise")

The variable p now represents the plugin. Plugins expose a number of properties, these are variables that can be inspected or set. For example the add noise plugin has a property called Sigma. To set this to a particular value we would use: 

.. code-block:: python

   p.Sigma = 0.5

Likewise, properties can be inspected: 

.. code-block:: python

   print(p.Sigma)
   
To list all propreties in a plugin use the method listOfProperties. The following uses call together with pprint to make the output more understandable: 

.. code-block:: python

   p.listOfProperties()
   [['NoiseType', 'Type of noise (Gaussian = 0).'], ['Sigma', 'Size of applied noise'], ['InputData', 'Data on which noise will be applied to'], ['AssumeTime', "Assume the first column in the input data is time, and don't add noise to it."], ['Progress', 'Indicate progress in (0-100%)'], ['Seed', "Seed to use for each use of 'execute' (0 to ignore)"]]
   
To run a plugin so that it carries out its function, use the execute method: 

.. code-block:: python

   p.execute()

The results from an execute call will either be saved to a file or more likely via properties of the plugin. As a trivial example, consider a plugin called "test" that has three properties, x, y and z. When the execute() method is called the plugin will take the values stored in x and y, add them together and store the result in z. The following script illustrates how the plugin would be used from Python: 

.. code-block:: python

   import rrplugins as tel
   p = tel.Plugin("test")
   p.x = 1.2
   p.y = 4.5
   p.execute()
   print(p.z)

The following script illustrates a more substantial example using the add_noise plugin:

.. code-block:: python

   import roadrunner
   import rrplugins as tel

   rr = roadrunner.RoadRunner() 
   rr.load("path to any model. Eg. some models are provided in source code, like, sbml_test_0001")

   data = rr.simulate(0, 10, 50)

   noisePlugin = tel.Plugin("tel_add_noise") 

   d = tel.getDataSeries(data) 

   noisePlugin.InputData = d 

   noisePlugin.Sigma = 5.e-6 

   noisePlugin.execute()

   noisePlugin.InputData.plot()


DataSeries
==========

The plugin system supports a special data type called a Data Series. This is a convenient way to represent rows and colums of data. The data type also has the ability to label columns with a text string and to associate each value in the data series with an additional value called a weight. In practice the data series will usually store experimental data and the weights will represent a measure of undertaintly, perhaps a standard deviation, of the data point. A Data Series can be created using the call: 

.. code-block:: python

   import rrplugins as tel
   data = tel.DataSeries()

Data can be entered into a data series either by loading the data from a specially formated file or from a Python NumPy array. For example: 

.. code-block:: python

   data.readDataSeries ("mydata.txt")

To read numpy arrays into a data series use the code: 

.. code-block:: python

   import numpy as np
   values = np.array([[1,2],{2,5],[3,7]])
   data = tel.DataSeries.fromNumPy (values)

The number of rows and columns in a Data Series can be obtained using: 

.. code-block:: python

   print(data.rows)
   print(data.cols)

Currently individual values in a data series can be accessed using the set and get methods: 

.. code-block:: python

   p.setElement (1, 2, 4.567)
   print(p.getElement(1, 2))

Data series can be plotted using the plot method: 

.. code-block:: python
   
   data.plot()

The following script is an example of using the add_noise plugin. This plugin takes a data series and add a given amount of Guassian noise to all data except the data in the first column. 

.. code-block:: python
   
   import rrplugins as tel
   p = tel.Plugin ("tel_add_noise")
   p.viewManual()
   pl = p.listOfProperties()
   for item in pl:
       print(item)
   
   p.Sigma = 0.00005
   series = DataSeries.loadDataSeries("..\\Examples\\testData.dat")
   series.plot()
   p.InputData = series
   p.execute()
   p.InputData.plot()
   
   print("Test Finished")


Plugins
=======

Plugin objects are instanciated using Plugin class. For example to instanciate a plugin called myplugin, we would use the code: 

.. code-block:: python
   
   p = Plugin("myplugin")
   
All interactions with plugins are via plugin properties. Values can be set and retrieved via plugin properties. For example, if a plugin has a property sigma, we can assign or access the value using the code: 

.. code-block:: python
   
   p.sigma = 0.1
   print(p.sigma)

Plugins have a single method that can be used to excecute the plugin's functionality: 

.. code-block:: python
   
   p.execute()

Once a plugin has been executed, any output from the plugin can be retrieved via properties. Let's suppose for example there is a plugin all "test", which has three properties called, x, y and result. When executed the plugin will take the values in x and y, compute the sum and assign it to result. The plugin can therefore be used as follows:
   
.. code-block:: python
   
   p = Plugin("add")
   p.x = 3.4
   p.y = 5.6
   p.execute()
   print(p.result)



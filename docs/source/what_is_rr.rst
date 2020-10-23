*********************
What is libRoadRunner
*********************
RoadRunner is a package for loading, simulating and
analyzing SBML based systems biology models utilzing JIT compilation.

RoadRunner 1.4.3

Up to date documentation can be found on http://libroadrunner.org/.

Also `documentation home page <../index.html>`_ provides an introduction.

Licence and Copyright
---------------------
libRoadRunner is free and open source. Licensing and Distribution 
Terms can be found in the LICENCE.txt file in the root directory 
of the distribution.

Copyright (C) 2012-2016 University of Washington, Seattle, WA, USA

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0.html
 
http://libroadrunner.org/

Fundamental Objects
-------------------
The libRoadRunner package uses two fundametal objects e.g. ``rr`` of 
class ``RoadRunner`` and e.g. ``rr.model`` of class ``ExecutableModel``. 

**RoadRunner**
 - Typically the top level object
 - Responsible for orchestrating all of the internal components, such as model loading, 
   JIT compilation, integration and output.
 - Initialized with ``rr = roadrunner.RoadRunner()``


**ExecutableModel**
 - Represents a compiled sbml model
 - Properties to get and set any state variables.
 - Initialized when SBML is loaded ``r.load('mymodel.xml')``

The Python API is a very clean simple interface that uses all native Python objects. 
All the returned types are structured `Numpy` arrays. 


Example of libRoadRunner in Use
-------------------------------
Transcript from an Python session to demonstrate libRoadRunner use on this interactive Python console.
 

**Import** roardrunner and numpy::

   import roadrunner
   import roadrunner.testing
   import numpy as n
   import numpy.linalg as lin

**Load** an SBML model::

   >>>  rr = roadrunner.RoadRunner()
   >>>  rr.load(roadrunner.testing.get_data('Test_1.xml'))
   True

Get the **model**, the model object has all the accessors sbml elements, names, values::
   
   >>>  m = rr.getModel()

Use the built in RR function to get the **Jacobian**, notice this is returned as a native
numpy matrix, and display it::
   
   >>>  jac = rr.getFullJacobian()
   >>>  jac
   array([[-0.2  ,  0.067,  0.   ],
          [ 0.15 , -0.467,  0.09 ],
          [ 0.   ,  0.4  , -0.64 ]])

Get a vector of **floating species amounts**, and display it::

   >>>  amt = m.getFloatingSpeciesAmounts()
   >>>  amt
   array([ 0.1 ,  0.25,  0.1 ])

Look at the **floating species ids**::
   
   >>>  m.getFloatingSpeciesIds()
   ['S1', 'S2', 'S3']

Numpy has a huge amount of numeric capability, here we calculate
the **eigensystem from the Jacobian**.::

   >>>  lin.eig(jac)
   (array([-0.15726345, -0.38237134, -0.76736521]),
    array([[ 0.77009381, -0.19510707,  0.03580588],
           [ 0.49121122,  0.53107368, -0.30320915],
           [ 0.40702219,  0.82455683,  0.95225109]]))

Suppose we wanted to calculate the matrix vector product of the **jacobian with the 
floating species amounts**, its a single statement, since we use native types.::

   >>>  n.dot(jac, amt)
   array([-0.00325, -0.09275,  0.036  ])

Finally, you can of course **simulate over time**. The first column in result is time, 
the rest are whatever is selected. The easies way to plot is to use :meth:`RoadRunner.plot`::
   
   >>>  results = rr.simulate()
   >>>  rr.plot(results)
   
.. seealso:: :ref:`plotting-data`
   
Using libRoadRunner in `IPython <http://ipython.org/>`_ you can **get documentation** 
easily using a ``?`` after the object or method::

  >>>  r.plot?

  Type:        instancemethod
  String form: <bound method RoadRunner.plot of <roadrunner.RoadRunner() { this = 0x101c70a00 }>>
  File:        /Users/andy/Library/Python/2.7/lib/python/site-packages/roadrunner/roadrunner.py
  Definition:  r.plot(self, show=True)
  Docstring:
  RoadRunner.plot([show])
  
  Plot the previously run simulation result using Matplotlib.
  
  This takes the contents of the simulation result and builds a
  legend from the selection list.
  
  
  If the optional prameter 'show' [default is True] is given, the pylab
  show() method is called.


Technical Footnotes
-------------------

.. [#] Most of the time, Numpy array holds a pointer to a block of data owned
       by RoadRunner. For example, the array returned by `rr.simulate()` has a pointer 
       to the results matrix which is owned by the `RoadRunner`, therefore NO COPYING 
       is involved. If you have no need for the result, simply ignore it, since it costs virtually nothing to return it. 
       
.. [#] Current State of the System Group
       
       When using the LLVM back end, all model state calculation are automatically 
       performed using a techinque called lazy evaluation. If one sets the concentration 
       of a specie, the amount of of that specie is automatically available without 
       having to perform any addition operations, similar to any other value in the model. 
       If an SBML parameter is defined by an assigment rule or a function and its value  
       depends on a number of other values, simply setting to other values automatically 
       cause the value of the most dependent variable to be set. 
       This is identical how one operates in a spredsheet such as Microsoft Excel. For 
       example, if one has a cell with an equation that depends on other cell, and those
       other cell depend on other values, setting the value of any upstream cell automatically
       causes that value to cascade down to the terminal cells. The LLVM back end roadruner
       function identically. 

.. highlight:: python

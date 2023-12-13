Solver Class
_________________________

.. class:: roadrunner.Solver

   Following functions are supported for a given solver instance which can be obtained by calling RoadRunner.getIntegrator() or RoadRunner.getSteadyStateSolver() methods. There are two types of solvers: Integrators(cvode, gillespie, rk4, rk45) - RoadRunner.Integrator() and Steady State Solvers(nleq2) - RoadRunner.SteadyStateSolver()


.. method:: Solver.getDescription(key)

   If key = None,  returns description of this solver.
   Else, returns the description associated with a given key.
   
   :param str key: settings name

.. method:: Solver.getDisplayName(key)

   Returns display name of given key.
   
   :param str key: settings name   


.. method:: Solver.getHint(key)

   If key = None,  returns a (user-readable) hint for this solver.
   Else, Returns the hint associated with a given key.
   
   :param str key: settings name
   

.. method:: Solver.getName()

   Returns the name of this solver


.. method:: Solver.getNumParams()

   Get the number of parameters.


.. method:: Solver.getParamDesc(n)

   Get the description of the parameter at index n

   :param int n: index of parameter


.. method:: Solver.getParamDisplayName(n)

    Get the display name of the parameter at index n

    :param int n: index of parameter


.. method:: Solver.getParamHint(n)

   Get the hint of the parameter at index n.
   
   :param int n: index of parameter
   

.. method:: Solver.getParamName(n)

   Get the display name of the parameter at index n.
   
   :param int n: index of parameter
   

.. method:: Solver.getSetting(k)

    Get value of a solver setting   

    :param str k: settings name


.. method:: Solver.getSettings()

   Returns a list of all settings for this solver.


.. method:: Solver.getSettingsRepr()

   Returns the solver settings as a string.


.. method:: Solver.getType(key)

   Returns the type associated with a given key.
   
   :param str key: settings name 

   
.. method:: Solver.getValue(key)

   Get the value of an integrator setting.
   
   :param str key: settings name

   
.. method:: Solver.getValueAsBool(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   
   

.. method:: Solver.getValueAsChar(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   
   

.. method:: Solver.getValueAsDouble(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   

   
.. method:: Solver.getValueAsFloat(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   
   

.. method:: Solver.getValueAsInt(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name
   
   
.. method:: Solver.getValueAsLong(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name

   
.. method:: Solver.getValueAsString(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name      
   
   
.. method:: Solver.getValueAsUChar(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   
  
   
.. method:: Solver.getValueAsUInt(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name   
   
   
.. method:: Solver.getValueAsULong(key)

   Wrapper for :meth:`Solver.getValue()` which converts output to a specific type
   
   :param str key: settings name
   
   
.. method:: Solver.hasValue(key)

   Return true if this setting is supported by the integrator.
   
   :param str key: settings name
   
   
.. method:: Solver.resetSettings()

   Reset all settings to their respective default values.


.. method:: Solver.setSetting(k, v)

    Set value of a solver setting   

    :param str k: settings name
    :param const v: value of the setting

   
.. method:: Solver.setValue(key, value)

   Sets value of a specific setting.
   
   :param str key: settings name    
   :param const value: value of the setting


.. method:: Solver.settingsPyDictRepr()

   Returns Python dictionary-style string representation of settings.
   

.. method:: Solver.syncWithModel(m)

    Called whenever a new model is loaded to allow integrator to reset internal state

    :param object m: new model
   

.. method:: Solver.toRepr()

   Return string representation a la Python __repr__ method.
   
   
.. method:: Solver.toString()

   Return a string representation of the solver.
   


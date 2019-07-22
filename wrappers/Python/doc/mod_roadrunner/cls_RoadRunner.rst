The Main RoadRunner Class
_________________________

.. class:: RoadRunner

   The main RoadRunner class. Its objects, i.e. ``rr = RoadRunner()`` perform 
   the libRoaRunner functions, i.e. ``rr.simulate()``.



.. method:: RoadRunner.__init__(uriOrSBML = "", options = None)

   Creates a new RoadRunner object. If the first argument is specified, 
   it should be a string containing either the contents of an SBML document, 
   or a formatted URI specifying the path or location of a SBML document. 

   If options is given, it should be a LoadSBMLOptions object. 

   If no arguments are given, a document may be loaded at any future time
   using the load method. 

   :param uriOrSBML: a URI, local path or SBML document contents.
   :type name: str (optional)

   :param options: (LoadSBMLOptions) an options object specifying how the 
                   SBML document should be loaded
   :type name: str (optional)


.. method:: RoadRunner.load(uriOrDocument)
   :module: RoadRunner

   Loads an SBML document, given a string for file path, URI, or contents. 

   This method also accepts HTTP URI for remote files, however this feature is currently limited 
   to the Mac version, plan on enabling HTTP loading of SBML documents on Windows and Linux 
   shortly. 

   Some examples of loading files on Mac or Linux::
   
       >>> r.load("myfile.xml")                               # load a file from the current directory
       >>> r.load("/Users/Fred/myfile.xml")                   # absolute path
       >>> r.load("http://sbml.org/example_system.xml")       # remote file


   Or on Windows:

       >>> r.load("myfile.xml")                                  # load a file from the current directory
       >>> r.load("file://localhost/c:/Users/Fred/myfile.xml")   # using a URI

   One may also load the contents of a document::

       >>> myfile = open("myfile.xml, "r")
       >>> contents = file.read()
       >>> r.load(contents)
       
   Loading in a raw SBML string is also possible:

       >>> sbmlstr = rr.getCurrentSBML()         # Or any other properly formatted SBML string block
       >>> r.load(sbmlstr)

   In future version, we will also support loading directly from a libSBML Document object. 

   :param uriOrDocument: A string which may be a local path, URI or contents of an SBML document. 
   :type name: str






.. method:: RoadRunner.getCompiler()

   Returns the JIT :class:`Compiler` object currently being used. 
   This object provides various information about the current processor and system.



.. method:: RoadRunner.getConfigurationXML()
   :module: RoadRunner

   Recurse through all of the child configurable objects that this
   class ones and build an assemble all of their configuration parameters
   into a single xml document which is returned as a string.

   The value of this result depends on what child objects are presently loaded.


.. method:: RoadRunner.getExtendedVersionInfo()
   :module: RoadRunner

   Returns :meth:`getVersionStr()` as well as info about dependent libs versions.



.. method:: RoadRunner.getInfo()
   :module: RoadRunner

   Returns info about the current state of the object.

   :rtype: str



.. method:: RoadRunner.getInstanceCount()
   :module: RoadRunner

   Returns number of currently running RoadRunner instances.



.. method:: RoadRunner.getInstanceID()
   :module: RoadRunner

   Returns the instance id when there are multiple instances of RoadRunner.



.. method:: RoadRunner.getIntegrator()
   :module: RoadRunner

   Returns the solver instance. See :class:`roadrunner.Solver`.
   For more information on the possible settings, see :ref:`roadrunner-solver`.
   
   
.. method:: RoadRunner.getIntegratorByName(name)
   :module: RoadRunner
   
   Returns the solver instance by given name. See :class:`roadrunner.Solver`.
   For more information on the possible settings, see :ref:`roadrunner-solver`.

   :param str name: Name of the integrator

   
.. method:: RoadRunner.getAvailableIntegrators()
   :module: RoadRunner
   
   Returns a list of names of available integrators.

   
.. method:: RoadRunner.getExistingIntegratorNames()
   :module: RoadRunner
   
   Returns a list of names of all integrators.
   

.. method:: RoadRunner.getParamPromotedSBML(*args)
   :module: RoadRunner

   Takes an SBML document (in textual form) and changes all of the local parameters
   to be global parameters.

   :param str SBML: the contents of an SBML document
   :rtype: str


.. method:: RoadRunner.getCurrentSBML()
   :module: RoadRunner

   Returns the SBML with the current model parameters. 
   This is different than :meth:`getSBML()` which returns the original SBML.
   This may optionally up or down-convert the document to a different version, if the level and version arguments are non-zero.
   If both arguments are zero, then the document is left alone and the original version is returned.

   :rtype: str


.. method:: RoadRunner.getSBML()
   :module: RoadRunner

   Returns the original SBML model that was loaded into roadrunner.
   If the model is edited by methods in editing section, it will return the most updated model with 
   the initial model parameters.

   :rtype: str
   
.. method:: RoadRunner.setIntegrator(name)
   :module: RoadRunner

   Sets specific integrator. For more information on the possible settings, see :ref:`roadrunner-solver`.
   
   :param str name: name of the integrator.

   
.. method:: RoadRunner.setIntegratorSetting(name, key, value)
   :module: RoadRunner

   Sets settings for a specific integrator. See :class:`roadrunner.Solver`.
   For more information on the possible settings, see :ref:`roadrunner-solver`.
   
   :param str name: name of the integrator.
   :param str key: name of the setting.
   :param const value: value of the setting.
   

Selections
----------

.. method:: RoadRunner.getIds()

   Return a list of selection ids that this object can select on.

   :rtype: list


.. method:: RoadRunner.getValue(sel)

   Returns the value for a given selection. For more information on accepted selection types
   see :ref:`Selecting Values <selecting-values>`.

   :param sel: a selection that is either a string or a SelectionRecord that was
               obtained from createSelection
   :type sel: str or SelectionRecord



.. method:: RoadRunner.getSelectedValues()
   :module: RoadRunner

   returns the values of the current timecourse selections for the current state of the model

   :rtype: numpy.ndarray



.. attribute:: RoadRunner.timeCourseSelections
   :module: RoadRunner
            
   Get or set the list of current selections used for the time course simulation result columns. 
   For more information on accepted selection types, see :ref:`Selecting Values <selecting-values>`.
   
   >>> rr.timeCourseSelections = ['time', 'S1', 'S2']
   >>> rr.timeCourseSelections
   ['time', 'S1', 'S2']


.. method:: RoadRunner.createSelection(sel)

   Create a new selection based on a selection string

   :rtype: roadrunner.SelectionRecord


.. method:: RoadRunner.resetSelectionLists()

   Resets time course and steady state selection lists to defaults

   
   
Model Access
------------


.. method:: RoadRunner.isModelLoaded()
   :module: RoadRunner

   Return True if model was loaded; False otherwise



.. py:attribute:: RoadRunner.model
   :module: RoadRunner
   :annotation: None

   Get the currently loaded model. The model object contains the entire state of the SBML model.


.. method:: RoadRunner.getModel()
   :module: RoadRunner

   Function form of the RoadRunner.model property, identical to model. 

   
.. method:: RoadRunner.clearModel()
   :module: RoadRunner
   
   Clears the currently loaded model and all associated memory.
   Returns True if memory was freed, False if no model was loaded in the first place.
   
   >>> r.isModelLoaded()
   True
   >>> r.clearModel()
   >>> r.isModelLoaded()
   False
   
   
.. method:: RoadRunner.oneStep(startTime, stepSize)
   :module: RoadRunner

   Carry out a one step integration of the model. The method takes two arguments,
   the current time and the step size to use in the integration. 
   Returs the new time which will be currentTime + StepSize::
       newTime = rr.oneStep (10, 0.5)

.. method:: RoadRunner.internalOneStep(startTime, stepSize, reset)
   :module: RoadRunner

   Carry out a single internal solver integration step. The method takes three arguments,
   the current time and the step size to use in the integration and reset. Reset defaults to true, set to false to stop integrator instance from restarting. Returns the end time.

.. method:: RoadRunner.reset()
   :module: RoadRunner

   Resets time, all floating species, and rates to their initial values.
   Does NOT reset changed global parameters.
   

.. method:: RoadRunner.resetAll()
   :module: RoadRunner

   Resets time, all floating species, and rates to their CURRENT initial values.
   Also resets all global parameters back to the values they had when the model was first loaded.
   "Current" initial values are set by using ``r.setValue('init(S1)', 5)`` which sets a species 
   named S1 to have current initial value of 5. Note it is NOT the initial values of when the model was first loaded in.

.. method:: RoadRunner.resetParameter()
   :module: RoadRunner

   Resets only global parameters to their CURRENT initial values.

.. method:: RoadRunner.resetToOrigin()
   :module: RoadRunner

   Resets the model back to the state it was when FIRST loaded.
   The scope of this reset includes all initial values and parameters (everything).

.. method:: RoadRunner.setConfigurationXML(*args)
   :module: RoadRunner

   Given a xml document, which should have been returned from getConfigurationXML,
   this method recurses though all the child configurable elements and sets their
   configuration to the values specified in the document.

   :param str xml: the contents of an xml document.



.. attribute:: RoadRunner.conservedMoietyAnalysis
   :module: RoadRunner

   Enables / Disables conserved moiety analysis (boolean). 

   If this is enabled, the SBML document (either current, or one about to be loaded) 
   is converted using the ConservedMoietyConverter. All of the linearly dependent 
   species are replaced with assignment rules and a new set of conserved moiety 
   parameters are introduced. 
   
   To enable, type:
   
   >>> r.conservedMoietyAnalysis = True
   
   
Model Editing
------------

Easy edit to the model without modifying and reloading sbml files.

.. method:: RoadRunner.addSpecies(sid, compartment, initValue, substanceUnits, forceRegenerate)
   :module: RoadRunner
   
   Add a species to the current model. Note that the species to be added must have an ID
   that did not existed in the model. The given compartment must also exist in the model.
   
   Default substanceUnits is "concentration", which will set initValue as initial concentration
   of the new species. Other substanceUnits will set initValue as initial amount of the new species.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addSpecies("s1", "compartment", 0.1, "concentration", false) # it will not regenerate the model, nothing actually happened
   >>> r.addSpecies("s2", "compartment", 0.1, "concentration", true)  # new model is generated and saved
  
   :param str sid: the ID of the species to be added
   :param str compartment: the compartment of the species to be added
   :param double initValue: the initial amount or concentration of the species to be added
   :param str substanceUnits: the substance unit of the species to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call

   
.. method:: RoadRunner.removeSpecies(sid, forceRegenerate)
   :module: RoadRunner
   
   Remove a species from the current model. Note that the given species must exist in the
   current model.
   
   All reactions related to this species(as reactants, products or modifiers or used in 
   stoichiometry) will be removed.
   Kinetic law used this species in the math formula will be unset.
   All function definitions, constraints, initial assignments and rules related to this species
   (as variables or used in math formula) will be removed.
   All events used this speices in trigger formula will be removed.
   Priority and delay used this sepcies in the math formula will be unset.
   All event assignment related to this species(as variables or used in math formula) will be removed.
   
   If any global parameters become uninitialized during this process, i.e, has no initial assignment or
   assignment rule, they will be removed recursively following the rules in removeParameter().
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.removeSpecies("s1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeSpecies("s2", true)  # new model is generated and saved

   :param str sid: the ID of the species to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
 
   
.. method:: RoadRunner.addReaction(rid, reactants, products, kineticLaw, forceRegenerate)
   :module: RoadRunner
   
   Add a reaction to the current model by passing its info as parameters. 
   Note that the reaction to be added must have an ID that did not existed in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addReaction("r1", ["s1"], ["s2"], "s1 * k1", false) # it will not regenerate the model, nothing actually happened
   >>> r.addReaction("r2", ["s2"], ["s1"], "s2 * k1", true)  # new model is generated and saved
  
  
   :param str rid: the ID of the reaction to be added
   :param list reactants: the list of reactants ID of reaction to be added
   :param list products: the list of products ID of reaction to be added
   :param str kineticLaw: the kinetic formular of reaction to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   

.. method:: RoadRunner.addReaction(sbmlRep, forceRegenerate)
   :module: RoadRunner
   
   Add a reaction to the current model by passing a sbml repesentation as parameter. 
   Note that the reaction to be added must have an ID that did not existed in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
  
   :param str sbmlRep: the SBML representation (i.e. a reaction tag) describing the reaction to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeReaction(rid, forceRegenerate)
   :module: RoadRunner
   
   Remove a reaction from the current model. Note that the given reaction must exist in the
   current model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.removeReaction("r1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeReaction("r2", true)  # new model is generated and saved

   :param str rid: the ID of the reaction to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addParameter(pid, value, forceRegenerate)
   :module: RoadRunner
   
   Add a parameter to the current model. Note that the parameter to be added must have an ID
   that did not existed in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addParameter("p1", 0.1, false) # it will not regenerate the model, nothing actually happened
   >>> r.addParameter("p2", 0.1, true)  # new model is generated and saved
  
   :param str pid: the ID of the parameter to be added
   :param double value: the initial value of the parameter to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeParameter(pid, forceRegenerate)
   :module: RoadRunner
   
   Remove a parameter from the current model. Note that the given parameter must exist in the
   current model.
   
   All reactions related to this parameter(used in stoichiometry) will be removed.
   Kinetic law used this parameter in the math formula will be unset.
   All function definitions, constraints, initial assignments and rules related to this parameter
   (as variables or used in math formula) will be removed.
   All events used this parameter in trigger formula will be removed.
   Priority and delay used this parameter in the math formula will be unset.
   All event assignment related to this parameter(as variables or used in math formula) will be removed.
   
   If any global parameters become uninitialized during this process, i.e, has no initial assignment or
   assignment rule, they will be removed recursively following the above rules.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.removeParameter("p1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeParameter("p2", true)  # new model is generated and saved

   :param str pid: the ID of the parameter to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
   
.. method:: RoadRunner.addCompartment(cid, initVolume, forceRegenerate)
   :module: RoadRunner
   
   Add a compartment to the current model. Note that the compartment to be added must have an ID
   that did not existed in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addCompartment("c1", 0.1, false) # it will not regenerate the model, nothing actually happened
   >>> r.addCompartment("c2", 0.1, true)  # new model is generated and saved
  
   :param str cid: the ID of the compartment to be added
   :param double initVolume: the initial volume of the compartment to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeCompartment(cid, forceRegenerate)
   :module: RoadRunner
   
   Remove a compartment from the current model. Note that the given compartment must exist in the
   current model.
   
   All reactions related to this compartment(used in stoichiometry) will be removed.
   Kinetic law used this compartment in the math formula will be unset.
   All function definitions, constraints, initial assignments and rules related to this compartment
   (as variables or used in math formula) will be removed.
   All events used this compartment in trigger formula will be removed.
   Priority and delay used this compartment in the math formula will be unset.
   All event assignment related to this compartment(as variables or used in math formula) will be removed.
   
   If any global parameters become uninitialized during this process, i.e, has no initial assignment or
   assignment rule, they will be removed recursively following the rules in removeParameter().
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.removeCompartment("c1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeCompartment("c2", true)  # new model is generated and saved

   :param str cid: the ID of the compartment to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.setKineticLaw(rid, kineticLaw, forceRegenerate)
   :module: RoadRunner
   
   Set hte kinetic law for an existing reaction in the current model. 
   Note that given reaction must exist in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.setKineticLaw("r1", "s1 * k1", false) # it will not regenerate the model, nothing actually happened
   >>> r.setKineticLaw("r2", "s2 * k1", true)  # new model is generated and saved
  
  
   :param str rid: the ID of the reaction to be modified
   :param str kineticLaw: the kinetic formular of reaction to be set
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addAssignmentRule(vid, formula, forceRegenerate)
   :module: RoadRunner
   
   Add an assignment rule for a variable to the current model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addAssignmentRule("s1", "s1 * k1", false) # it will not regenerate the model, nothing actually happened
   >>> r.addAssignmentRule("s2", "s2 * k1", true)  # new model is generated and saved
  
  
   :param str vid: the ID of the variable that the new rule assigns formula to
   :param str formula: the math formula of assignment rule to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addRateRule(vid, formula, forceRegenerate)
   :module: RoadRunner
   
   Add a rate rule for a variable to the current model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addRateRule("s1", "k1", false) # it will not regenerate the model, nothing actually happened
   >>> r.addRateRule("s2", "k1", true)  # new model is generated and saved
  
  
   :param str vid: the ID of the variable that the new rule assigns formula to
   :param str formula: the math formula of rate rule to be added
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeRules(vid, forceRegenerate)
   :module: RoadRunner
   
   Remove all rules for a variable from the current model, including assignment and rate rules. 
   Note that the given variable must have at least one rule in the current model.
   
   If any global parameters become uninitialized during this process, i.e, has no initial assignment or
   assignment rule, they will be removed recursively following the rules in removeParameter().
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.removeRules("s1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeRules("s2", true)  # new model is generated and saved

   :param str vid: the ID of the variables that rules assign formula to
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addEvent(eid, useValuesFromTriggerTime, trigger, forceRegenerate)
   :module: RoadRunner
   
   Add an event to the current model. Note that the event to be added must have an ID that 
   did not existed in the model.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addEvent("e1", false, "s1 > 0", false) # it will not regenerate the model, nothing actually happened
   >>> r.addEvent("e2", false, "s2 == s1", true)  # new model is generated and saved
  
  
   :param str eid: the ID of the event to be added
   :param bool useValuesFromTriggerTime: indicate the moment at which the event's assignments are to be evaluated
   :param str trigger: the math formula of event trigger
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addTrigger(eid, trigger, forceRegenerate)
   :module: RoadRunner
   
   Add trigger to an existing event in the model. Note that the given event must exist in the
   current model. If the given event already has a trigger object, the given trigger will replace the 
   old trigger of the event.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
   For example,
   
   >>> r.addTrigger("e1", "s1 > 0", false) # it will not regenerate the model, nothing actually happened
   >>> r.addTrigger("e2", "s2 == s1", true)  # new model is generated and saved
  
  
   :param str eid: the ID of the event to add the trigger to
   :param str trigger: the math formula of event trigger
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addPriority(eid, priority, forceRegenerate)
   :module: RoadRunner
   
   Add priority to an existing event in the model. Note that the given event must exist in the
   current model. If the given event already has a priority object, the given priority will replace the 
   old priority of the event.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
  
   :param str eid: the ID of the event to add the priority to
   :param str priority: the math formula of event priority
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.addDelay(eid, delay, forceRegenerate)
   :module: RoadRunner
   
   Add delay to an existing event in the model. Note that the given event must exist in the
   current model. If the given event already has a delay object, the given delay will replace the 
   old delay of the event.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
  
   :param str eid: the ID of the event to add the delay to
   :param str delay: the math formula of event delay
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
 
.. method:: RoadRunner.addEventAssignment(eid, vid, formula, forceRegenerate)
   :module: RoadRunner
   
   Add an event assignment to an existing event in the model. Note that the given event must exist in the
   current model. 
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
    For example,
   
   >>> r.addEventAssignment("e1", "s1", "k1", false) # it will not regenerate the model, nothing actually happened
   >>> r.addEventAssignment("e2", "s2", "s1", true)  # new model is generated and saved
   
  
   :param str eid: the ID of the event to add the event assignment to
   :param str vid: the ID of the variables that assignment assigns formula to
   :param str formula: the math formula of event assignment
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeEventAssignment(eid, vid, forceRegenerate)
   :module: RoadRunner
   
   Add all event assignments for a variable from an existing event in the model. 
   Note that the given event must exist in the current model and given variable must have an event 
   assignment in the given event.
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
    For example,
   
   >>> r.removeEventAssignment("e1", "s1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeEventAssignment("e2", "s2", true)  # new model is generated and saved
   
  
   :param str eid: the ID of the event 
   :param str vid: the ID of the variables of the event assignments to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
   
   
.. method:: RoadRunner.removeEvent(eid, forceRegenerate)
   :module: RoadRunner
   
   Add an event from the current model. Note that the given event must exist in the current model.
   
   If any global parameters become uninitialized during this process, i.e, has no initial assignment or
   assignment rule, they will be removed recursively following the rules in removeParameter().
   
   forceRegenerate is a boolean value that indicates whether the new model will be regenerated. Its 
   default value is true, which means to regenerate model every time after this function is called. 
   Note that regenerating model is time-consuming. To save time for editing model for multiple times, 
   one could set this flag to false excepting for the last call, so that Roadrunner will only regenerate 
   the model once after all editings are completed.
   
    For example,
   
   >>> r.removeEvent("e1", false) # it will not regenerate the model, nothing actually happened
   >>> r.removeEvent("e2", true)  # new model is generated and saved
   
  
   :param str eid: the ID of the event to be removed
   :param bool forceRegenerate: indicate whether the new model is regenerated after this function call
 

Simulation
----------

Fast and easy time series simulations is one of the main objectives of the RoadRunner project. 

All simulation related tasks can be accomplished with the single ``simulate`` method. 
 

.. method:: RoadRunner.simulate(*args, **kwargs)
   :module: RoadRunner



   Simulate and optionally plot current SBML model. This is the one stop shopping method
   for simulation and plotting. 

   simulate accepts a up to four positional arguments. 

   The first four (optional) arguments are treated as:
            
      1: Start Time, if this is a number. 

      2: End Time, if this is a number.

      3: Number of points, if this is a number.
            
      4: List of Selections. A list of variables to include in the output, e.g. ``['time','A']`` for a model with species ``A``. More below.

   All four of the positional arguments are optional. If any of the positional arguments are
   a list of string instead of a number, then they are interpreted as a list of selections. 
   
   There are a number of ways to call simulate.

   1: With no arguments. In this case, the current set of options from the previous 
      ``simulate`` call will be used. If this is the first time ``simulate`` is called, 
      then a default set of values is used. The default set of values are (start = 0, end = 5, points = 51).

   2: With up to four positions arguments, described above. 
   
   Finally, you can pass steps keyword argument instead of points. 
   
   steps (Optional) Number of steps at which the output is sampled where the samples are evenly spaced. Steps = points-1. Steps and points may not both be specified.

   :returns: a numpy array with each selected output time series being a
             column vector, and the 0'th column is the simulation time.
   :rtype: numpy.ndarray

   
.. method:: RoadRunner.gillespie(start, end, steps)
   :module: RoadRunner
   
   Run a Gillespie stochastic simulation.
   
   Use :meth:`RoadRunner.reset()` to reset the model each time.
   
   :param start: start time
   :param end: end time
   :param steps: number of steps
   :returns: a numpy array with each selected output time series being a
             column vector, and the 0'th column is the simulation time.
   :rtype: numpy.ndarray
   
   Examples:

   Simulate from time zero to 40 time units 
   
   >>> result = r.gillespie (0, 40)

   Simulate on a grid with 10 points from start 0 to end time 40 
   
   >>> result = r.gillespie (0, 40, 10)

   Simulate from time zero to 40 time units using the given selection list 
   
   >>> result = r.gillespie (0, 40, [‘time’, ‘S1’])

   Simulate from time zero to 40 time units, on a grid with 20 points using the given selection list 
   
   >>> result = r.gillespie (0, 40, 20, [‘time’, ‘S1’])
   

.. py:function:: RoadRunner.plot(result=None, loc='upper left', show=True)
   :module: RoadRunner
   
   Plot results from a simulation carried out by the simulate or gillespie functions. 
  
   To plot data currently held by roadrunner that was generated in the last simulation, use:
   
   >>> r.plot() 
   
   If you are using Tellurium, see `tellurium.ExtendedRoadRunner.plot <https://tellurium.readthedocs.io/en/latest/tellurium_methods.html#tellurium.tellurium.ExtendedRoadRunner.plot>`_ which supports extra arguements.
   
   :param numpy.ndarray result: Data returned from a simulate or gillespie call
   :param str loc: string representing the location of legend i.e. "upper right"


.. py:function:: Roadrunner.getSimulationData()
   :module: RoadRunner

   Returns the array of simulated data. When simulation has not been run,
   the function will return an empty array.



Steady State
---------------------

.. class:: RoadRunner.SteadyStateSolver
   :module: RoadRunner
   
   RoadRunner.SteadyStateSolver class.


.. attribute:: RoadRunner.steadyStateSelections
   :module: RoadRunner

   A list of SelectionRecords which determine what values are used for 
   a steady state calculation. This list may be set by assigning a list
   of valid selection symbols::

     >>> r.steadyStateSelections = ['S1', '[S2]', 'P1']
     >>> r.steadyStateSelections
     ['S1', '[S2]', 'P1']


.. method:: RoadRunner.steadyState()
   :module: RoadRunner

   Attempts to evaluate the steady state for the model. The method returns
   a value that indicates how close the solution is to the steady state.
   The smaller the value the better. Values less than 1E-6 usually indicate a
   steady state has been found. If necessary the method can be called a
   second time to improve the solution.

   :returns: the sum of squares of the steady state solution.

   :rtype: double


.. method:: RoadRunner.getSteadyStateValues()
   :module: RoadRunner
    
   Performs a steady state calculation (evolves the system to a steady
   state), then calculates and returns the set of values specified by
   the steady state selections.

   :returns: a numpy array corresponding to the values specified by steadyStateSelections

   :rtype: numpy.ndarray

   
.. method:: RoadRunner.getSteadyStateValuesNamedArray()
   :module: RoadRunner
    
   Performs a steady state calculation (evolves the system to a steady
   state), then calculates and returns the set of values specified by
   the steady state selections with all necessary labels.

   :returns: a NamedArray corresponding to the values specified by steadyStateSelections

   :rtype: NamedArray
   

.. method:: RoadRunner.getSteadyStateSolver()
   :module: RoadRunner
   
   Returns the steady state solver which is currently being used.   


.. method:: RoadRunner.steadyStateSolverExists(name)
   :module: RoadRunner   
   
   Checks whether a steady state solver exists.
   
   :param str name: name of a steady state solver



Metabolic control analysis
--------------------------

In the special case when an SBML model is a purely reaction kinetics model -- no rate rules, no
assignment rules for chemical species, and time invariant stoichiometry, specialized analysis methods
related to metabolic control analysis are applicable. These methods are described in this section. 


.. method:: RoadRunner.getCC(variable, parameter)
   :module: RoadRunner

   Returns a scaled control coefficient with respect to a global parameter.
   
   For example::

     rr.getCC ('J1', 'Vmax')
     rr.getCC ('S1', 'Xo')
     rr.getCC ('S2', 'Km')

   The first returns a flux control coefficient with respect to flux J1. The second and third
   return concentration control coefficients with respect to species S1 and S2.

   :param variable: The id of a dependent variable of the coefficient, for example a
                    reaction or species concentration.

   :param parameter: The id of the independent parameter, for example a kinetic constant
                     or boundary species

   :returns: the value of the scaled control coefficient.

   :rtype: double


.. method:: RoadRunner.getuCC(variableId, parameterId)
   :module: RoadRunner

   Get unscaled control coefficient with respect to a global parameter.

   :param variableId: The id of a dependent variable of the coefficient, for example a
                    reaction or species concentration.

   :param parameterId: The id of the independent parameter, for example a kinetic constant
                     or boundary species
					 
   :returns: the value of the unscaled control coefficient.

   :rtype: double
					   
.. method:: RoadRunner.getEE(reactionId, parameterId, steadyState=True)
   :module: RoadRunner

   Retrieve a single elasticity coefficient with respect to a global parameter.
   
   For example::

     x = rr.getEE ('J1', 'Vmax')
	 
   calculates elasticity coefficient of reaction 'J1' with restpect to parameter 'Vmax'.

   :param str variable: A reaction Id
   :param str parameter: The independent parameter, for example a kinetic constant, floating or boundary species
   :param Boolean steadyState: should the steady state value be computed.


.. method:: RoadRunner.getuEE(reactionId, parameterId)
   :module: RoadRunner

   Get unscaled elasticity coefficient with respect to a global parameter or species.


.. method:: RoadRunner.getEigenValueIds()
   :module: RoadRunner

   Returns a list of selection symbols for the eigenvalues of the floating species. The eigen value
   selection symbol is ``eigen(XX)``, where ``XX`` is the floating species name. 

   
.. method:: RoadRunner.getFullEigenValues()
   :module: RoadRunner


   Calculates the eigen values of the Full Jacobian as a real matrix, first column real part, second
   column imaginary part.

   Note, only valid for pure reaction kinetics models (no rate rules, no floating species rules and
   time invariant stoichiometry). 

   :rtype: numpy.ndarray


.. method:: RoadRunner.getReducedEigenValues()
   :module: RoadRunner


   Calculates the eigen values of the Reduced Jacobian as a real matrix, first column real part, second
   column imaginary part.

   Only valid if moiety conversion is enabled. 

   Note, only valid for pure reaction kinetics models (no rate rules, no floating species rules and
   time invariant stoichiometry). 

   :rtype: numpy.ndarray


.. method:: RoadRunner.getFullJacobian()
   :module: RoadRunner

   Compute the full Jacobian at the current operating point.

   This is the Jacobian of ONLY the floating species.


.. method:: RoadRunner.getReducedJacobian()
   :module: RoadRunner

   Returns the *reduced* Jacobian for the independent species. This matrix will be non-singular
   for models that include moiety-conserved cycles.

   :rtype: numpy.ndarray




.. method:: RoadRunner.getScaledConcentrationControlCoefficientMatrix()
   :module: RoadRunner

   Returns the m by n matrix of scaled concentration control coefficients where m is the number
   of floating species and n the number of reactions.

   :rtype: numpy.ndarray


.. method:: RoadRunner.getScaledFloatingSpeciesElasticity(reactionId, speciesId)
   :module: RoadRunner

   Returns the scaled elasticity for a given reaction and given species.

   :param str reactionId: the SBML id of a reaction.
   :param str speciesId: the SBML id of a species.
   :rtype: double


.. method:: RoadRunner.getUnscaledParameterElasticity(reactionId, parameterId)
   :module: RoadRunner

   Returns the unscaled elasticity for a named reaction with respect to a
   named parameter
     
   :param str reactionId: the SBML id of a reaction.
   :param str parameterId: the SBML id of a parameter.
   :rtype: double


.. method:: RoadRunner.getUnscaledConcentrationControlCoefficientMatrix()
   :module: RoadRunner

   Returns the unscaled concentration control coefficient matrix.


.. method:: RoadRunner.getUnscaledElasticityMatrix()
   :module: RoadRunner

   Returns the unscaled species elasticity matrix at the current operating point.


.. method:: RoadRunner.getUnscaledFluxControlCoefficientMatrix()
   :module: RoadRunner

   Returns the unscaled flux control coefficient matrix.


.. method:: RoadRunner.getUnscaledSpeciesElasticity(reactionIndx, speciesIndx)
   :module: RoadRunner

   Get a single species elasticity value. 


   :param int reactionIndx: index of reaction
   :param int speciesIndx: index of species.


.. method:: RoadRunner.getScaledFluxControlCoefficientMatrix()
   :module: RoadRunner

   Returns the n by n matrix of scaled flux control coefficients where n is the number of reactions.

   :rtype: numpy.ndarray



.. method:: RoadRunner.getScaledElasticityMatrix()
   :module: RoadRunner

   Returns the scaled elasticity matrix at the current operating point.

   :rtype: numpy.ndarray
   

.. method:: RoadRunner.getDiffStepSize()
   :module: RoadRunner

   Returns the differential step size used in routines such as :meth:`getCC()`.
   

.. method:: RoadRunner.setDiffStepSize(val)
   :module: RoadRunner

   Sets the differential step size used in routines such as :meth:`getCC()`.
   
   :param val: differential step size
   
   
.. method:: RoadRunner.getSteadyStateThreshold()
   :module: RoadRunner
   
   Returns the threshold used in steady state solver in routines such as :meth:`getCC()`.


.. method:: RoadRunner.setSteadyStateThreshold(val)
   :module: RoadRunner
   
   Sets the threshold used in steady state solver in routines such as :meth:`getCC()`.   

   :param val: threshold value
   
   

Stoichiometric Analysis
-----------------------

.. method:: RoadRunner.getFullStoichiometryMatrix()
   :module: RoadRunner


   Get the stoichiometry matrix that coresponds to the full model, even it
   it was converted via conservation conversion.


.. method:: RoadRunner.getReducedStoichiometryMatrix()
   :module: RoadRunner

   Get the reduced stochiometry matrix. If conservation conversion is enabled,
   this is the matrix that coresponds to the independent species.

   A synonym for getNrMatrix().



.. method:: RoadRunner.getConservationMatrix()
   :module: RoadRunner

   Returns a conservation matrix :math:`\Gamma` which is a :math:`c \times m` matrix
   where :math:`c` is the number of conservation laws and :math:`m` the number of species.



.. method:: RoadRunner.getL0Matrix()
   :module: RoadRunner

   Returns the L0 matrix for the current model. The L0 matrix is an (m-r) by r matrix that expresses
   the dependent reaction rates in terms of the independent rates. m is the number of floating species
   and r is the rank of the stoichiometry matrix.

   :rtype: numpy.ndarray



.. method:: RoadRunner.getLinkMatrix()
   :module: RoadRunner

   Returns the full link matrix, L for the current model. The Link matrix is an m by r matrix where m
   is the number of floating species and r the rank of the stoichiometric matrix, N.

   :rtype: numpy.ndarray

.. method:: RoadRunner.getNrMatrix()
   :module: RoadRunner

   Returns the reduced stoichiometry matrix, :math:`N_R`, which will have only r rows where r is the rank of
   the full stoichiometry matrix. The matrix will be reordered such that the rows of :math:`N_R` are independent.

   :rtype: numpy.ndarray


.. method:: RoadRunner.getKMatrix()
   :module: RoadRunner

   Returns the K matrix, :math:`ker(N_R)`, (right nullspace of Nr)
   The K matrix has the structure, :math:`[I K0]'`

   :rtype: numpy.ndarray


Analysis
--------

.. method:: RoadRunner.getFrequencyResponse(startFrequency, numberOfDecades, numberOfPoints, parameterName, variableName, useDB, useHz)
   :module: RoadRunner

   Computes the frequency response.
   Returns a numpy array with three columns. First column is the frequency,
   second column the amplitude, and the third column the phase.

   :param startFrequency:  Start frequency for the first column in the output
   :param int numberOfDecades: Number of decades for the frequency range, eg 4 means the frequency span 10,000
   :param int numberOfPoints: The number of points to generate in the output
   :param str parameterName: The parameter where the input frequency is applied, usually a boundary species, eg 'Xo'
   :param str variableName: The amplitude and phase will be output for this variable, usually a floating species, eg 'S1'
   :param boolean useDB: If true use Decibels on the amplitude axis
   :param boolean useHz: If true use Hertz on the x axis, the default is rads/sec
    
   For example::
   
     import tellurium as te
     import roadrunner
     from matplotlib import pyplot as plt

     r = te.loada("""
         $Xo -> x1; k1*Xo - k2*x1;
          x1 -> x2; k2*x1 - k3*x2;
          x2 ->; k3*x2;

          k1 = 0.5; k2 = 0.23; k3 = 0.4;  Xo = 5;
     """)

     r.steadyState()

     m = r.getFrequencyResponse(0.001, 5, 1000, 'Xo', 'x2', True, False)

     fig = plt.figure(figsize=(10,4))

     ax1 = fig.add_subplot (121)
     ax1.semilogx (m[:,0], m[:,1], color="blue", linewidth="2")
     ax1.set_title ('Amplitude')
     plt.xlabel ('Frequency')

     ax2 = fig.add_subplot (122)
     ax2.semilogx (m[:,0], m[:,2], color="blue", linewidth="2")
     ax2.set_title ('Phase')
     plt.xlabel ('Frequency')
     plt.show()


.. method:: RoadRunner.getRatesOfChange()
   :module: RoadRunner

   Returns the rates of change of all floating species. The order of species is 
   given by the order of Ids returned by getFloatingSpeciesIds()

   :returns: a named array of floating species rates of change.
   :rtype: numpy.ndarray

   >>> r.getRatesOfChange()
             MKKK,       MKKK_P,      MKK,      MKK_P,    MKK_PP,     MAPK,   MAPK_P,  MAPK_PP
   [[ 0.000503289, -0.000503289, 0.537508, -0.0994839, -0.438024, 0.061993, 0.108417, -0.17041]]
   
   
.. method:: RoadRunner.getIndependentRatesOfChange()
   :module: RoadRunner

   Returns the rates of change of all independent floating species. The order of species is 
   given by the order of Ids returned by getIndependentFloatingSpeciesIds()

   :returns: a named array of independent floating species rates of change.
   :rtype: numpy.ndarray

   >>> r.getIndependentRatesOfChange()
           MKK_P,   MAPK_P,        MKKK,      MKK,     MAPK
   [[ -0.0994839, 0.108417, 0.000503289, 0.537508, 0.061993]]


.. method:: RoadRunner.getDependentRatesOfChange()
   :module: RoadRunner

   Returns the rates of change of all dependent floating species. The order of species is 
   given by the order of Ids returned by getDependentFloatingSpeciesIds()

   :returns: a named array of dependent floating species rates of change.
   :rtype: numpy.ndarray

   >>> r.getDependentRatesOfChange()
         MKK_PP,       MKKK_P,  MAPK_PP
   [[ -0.438024, -0.000503289, -0.17041]]
     

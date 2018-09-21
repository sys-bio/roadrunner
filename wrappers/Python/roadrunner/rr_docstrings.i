%feature("docstring") rr::ExecutableModel::keys "
ExecutableModel.keys()

Get a list of all the keys that this model has. This is a very good way of looking at all the
available symbols and selection strings:

>>>  r.model.keys()
[ 'S1', 'S2', '[S1]', '[S2]', 'compartment', 'k1', 'cm0',
  'reaction1',  'init([S1])',  'init([S2])', 'init(S1)',
  'init(S2)',  \"S1'\"]
";



%feature("docstring") rr::ExecutableModel::items "
ExecutableModel.items()

Get a list of key / value pairs of all the selections / values in this model.

.. method:: ExecutableModel.__getitem__

Implements the python ``[]`` indexing operator, so the model values can be accessed like::

  >>> r.model[\"S1\"]
  0.0

.. method:: ExecutableModel.__setitem__

Implements the python ``[]`` indexing operator for setting values::

  >>> r.model[\"S1\"] = 12.3

Note, some keys are read only such as values defined by rules, or calculated values such as
species amount rates or reaction rates. If one attempts to set the value of a read-only symbol,
an exception is raised indicating the error, and no harm done.



Floating Species
----------------
";



%feature("docstring") rr::ExecutableModel::getNumFloatingSpecies "
ExecutableModel.getNumFloatingSpecies()

Returns the number of floating species in the model.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesIds "
ExecutableModel.getFloatingSpeciesIds()

Return a list of floating species sbml ids.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesAmounts "
ExecutableModel.getFloatingSpeciesAmounts([index])

Get the list of floating species amounts. If no arguments are given, this
returns all floating species amounts.

:param index: an optional array of desired floating species indices. i.e. if this model has
              4 floating species and we want the amounts for the last and first, we
              would use [3,0] for the index array.
:type name: numpy.ndarray

get all the amounts::

  >>> e.getFloatingSpeciesAmounts()
  [15,2,3,20]

get amounts 3 and 0::

  >>> getFloatingSpeciesAmounts([3,0])
  [10,15]
";



%feature("docstring") rr::ExecutableModel::setFloatingSpeciesAmounts "
ExecutableModel.setFloatingSpeciesAmounts([index], values)

Use this to set the entire set of floating species amounts in one call.
The order of species is given by the order of Ids returned by getFloatingSpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be
                            an array of all the values to set.
:param numpy.ndarray values: the values to set.
";



%feature("docstring") rr::ExecutableModel::setFloatingSpeciesConcentrations "
ExecutableModel.setFloatingSpeciesConcentrations([index], values)

Use this to set the entire set of floating species concentrations in one call.
The order of species is given by the order of Ids returned by getFloatingSpeciesIds()


:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be an
                            array of all the  values to set.
:param numpy.ndarray values: the values to set.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesConcentrations "
ExecutableModel.getFloatingSpeciesConcentrations([index])

Returns a vector of floating species concentrations. The order of species is
given by the order of Ids returned by getFloatingSpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of floating species concentrations.
:rtype: numpy.ndarray
";


%feature("docstring") rr::RoadRunner::getRatesOfChange "
RoadRunner::getRatesOfChange()

Returns the rates of change of all floating species. The order of species is 
given by the order of Ids returned by getFloatingSpeciesIds()

:returns: a named array of floating species rates of change.
:rtype: numpy.ndarray


Floating Species Initial Conditions
-----------------------------------
RoadRunner stores all initial conditions separately from the model state variables. This means that
you can update the initial conditions at any time, and it does not affect the current state of the
model. To reset the model, that is, reset it to its original state, or a new original state where
what has changed the initial conditions use the :meth:`~ExecutableModel.reset()` method.

The following methods allow access to the floating species initial condition values:
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesInitAmountIds "
ExecutableModel.getFloatingSpeciesInitAmountIds()

get a list of the floating species amount initial amount selection symbols.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesInitConcentrationIds "
ExecutableModel.getFloatingSpeciesInitConcentrationIds()

get a list of the floating species amount initial concentration selection symbols.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesInitConcentrations "
ExecutableModel.getFloatingSpeciesInitConcentrations([index])

Returns a vector of floating species initial concentrations. The order of species is
given by the order of Ids returned by getFloatingSpeciesInitialConcentrationIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of floating species initial concentrations.
:rtype: numpy.ndarray
";



%feature("docstring") rr::ExecutableModel::setFloatingSpeciesInitConcentrations "
ExecutableModel.setFloatingSpeciesInitConcentrations([index], values)

Set a vector of floating species initial concentrations. The order of species is
given by the order of Ids returned by getFloatingSpeciesInitialAmountIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
";



%feature("docstring") rr::ExecutableModel::getFloatingSpeciesInitAmounts "
ExecutableModel.getFloatingSpeciesInitAmounts([index])

Returns a vector of floating species initial amounts. The order of species is
given by the order of Ids returned by getFloatingSpeciesInitialConcentrationIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of floating species initial amounts.
:rtype: numpy.ndarray
";



%feature("docstring") rr::ExecutableModel::setFloatingSpeciesInitAmounts "
ExecutableModel.setFloatingSpeciesInitAmounts([index], values)

Set a vector of floating species initial amounts. The order of species is
given by the order of Ids returned by getFloatingSpeciesInitialAmountIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.




Boundary Species
----------------
";



%feature("docstring") rr::ExecutableModel::getBoundarySpeciesAmounts "
ExecutableModel.getBoundarySpeciesAmounts([index])

Returns a vector of boundary species amounts. The order of species is
given by the order of Ids returned by getBoundarySpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of the boundary species amounts.
:rtype: numpy.ndarray.

given by the order of Ids returned by getBoundarySpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of the boundary species amounts.
:rtype: numpy.ndarray.
";



%feature("docstring") rr::ExecutableModel::getBoundarySpeciesConcentrations "
ExecutableModel.getBoundarySpeciesConcentrations([index])

Returns a vector of boundary species concentrations. The order of species is
given by the order of Ids returned by getBoundarySpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of the boundary species concentrations.
:rtype: numpy.ndarray.

given by the order of Ids returned by getBoundarySpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of the boundary species concentrations.
:rtype: numpy.ndarray.
";



%feature("docstring") rr::ExecutableModel::getBoundarySpeciesIds "
ExecutableModel.getBoundarySpeciesIds()

Returns a vector of boundary species Ids.

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: a list of boundary species ids.
";



%feature("docstring") rr::ExecutableModel::getNumBoundarySpecies "
ExecutableModel.getNumBoundarySpecies()

Returns the number of boundary species in the model.
";



%feature("docstring") rr::ExecutableModel::setBoundarySpeciesConcentrations "
ExecutableModel.setBoundarySpeciesConcentrations([index], values)

Use this to set the entire set of boundary species concentrations in one call.
The order of species is given by the order of Ids returned by getBoundarySpeciesIds()

:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be an
                            array of all the  values to set.

:param numpy.ndarray values: the values to set.

Compartments
------------
";



%feature("docstring") rr::ExecutableModel::getCompartmentIds "
ExecutableModel.getCompartmentIds([index])

Returns a vector of compartment identifier symbols.

:param index: A array of compartment indices indicating which compartment ids to return.
:type index: None or numpy.ndarray
:returns: a list of compartment ids.
";



%feature("docstring") rr::ExecutableModel::getCompartmentVolumes "
ExecutableModel.getCompartmentVolumes([index])

Returns a vector of compartment volumes. The order of volumes is
given by the order of Ids returned by getCompartmentIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of compartment volumes.
:rtype: numpy.ndarray.
";



%feature("docstring") rr::ExecutableModel::getNumCompartments "
ExecutableModel.getNumCompartments()

Returns the number of compartments in the model.

:rtype: int
";



%feature("docstring") rr::ExecutableModel::setCompartmentVolumes "
ExecutableModel.setCompartmentVolumes([index], values)

Sets a vector of compartment volumes.

If the index vector is not give, then the values vector treated as a vector of all
compartment volumes to set. If index is given, then  values should have the same
length as index.

:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be an
                            array of all the  values to set.

:param numpy.ndarray values: the values to set.



Global Parameters
-----------------
";



%feature("docstring") rr::ExecutableModel::getGlobalParameterValues "
ExecutableModel.getGlobalParameterValues([index])

Return a vector of global parameter values. The order of species is
given by the order of Ids returned by getGlobalParameterIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of global parameter values.
:rtype: numpy.ndarray.
";



%feature("docstring") rr::ExecutableModel::getNumGlobalParameters "
ExecutableModel.getNumGlobalParameters()


Returns the number of global parameters in the model.
";



%feature("docstring") rr::ExecutableModel::setGlobalParameterValues "
ExecutableModel.setGlobalParameterValues([index], values)

Use this to set the entire set of global parameters in one call.
The order of parameters is given by the order of Ids returned by getGlobalParameterIds()


:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be an
                            array of all the values to set.
:param numpy.ndarray values: the values to set.

Reactions
---------
";



%feature("docstring") rr::ExecutableModel::getNumReactions "
ExecutableModel.getNumReactions()

Returns the number of reactions in the model.
";



%feature("docstring") rr::ExecutableModel::getReactionIds "
ExecutableModel.getReactionIds()

Returns a vector of reaction Ids.

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: a list of reaction ids.
";



%feature("docstring") rr::ExecutableModel::getReactionRates "
ExecutableModel.getReactionRates([index])

Returns a vector of reaction rates (reaction velocity) for the current state of the model. The
order of reaction rates is given by the order of Ids returned by getReactionIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of reaction rates.
:rtype: numpy.ndarray



Stoichiometry
-------------
";



%feature("docstring") rr::ExecutableModel::getStoichiometry "
ExecutableModel.getStoichiometry(speciesIndex, reactionIndex)

Returns the stochiometric coefficient for the given species index and reaction index.

Frequently one does not need the full stochiometrix matrix, particularly if the system is
large and only a single coefficient is needed.


:param speciesIndex: a floating species index from :meth:`getFloatingSpeciesIds`
:param reactionIndex: a reaction index from :meth:`getReactionIds`


Conserved Moieties
------------------
";



%feature("docstring") rr::ExecutableModel::getNumConservedMoieties "
ExecutableModel.getNumConservedMoieties()

Returns the number of conserved moieties in the model.

:rtype: int
";



%feature("docstring") rr::ExecutableModel::getConservedMoietyIds "
ExecutableModel.getConservedMoietyIds([index])

Returns a vector of conserved moiety identifier symbols.


:param index: A array of compartment indices indicating which compartment ids to return.
:type index: None or numpy.ndarray
:returns: a list of compartment ids.
";



%feature("docstring") rr::ExecutableModel::getConservedMoietyValues "
ExecutableModel.getConservedMoietyValues([index])

Returns a vector of conserved moiety volumes. The order of values is
given by the order of Ids returned by getConservedMoietyIds()

:param numpy.ndarray index: (optional) an index array indicating which items to return.
:returns: an array of conserved moiety values.
:rtype: numpy.ndarray.
";



%feature("docstring") rr::ExecutableModel::setConservedMoietyValues "
ExecutableModel.setConservedMoietyValues([index], values)

Sets a vector of conserved moiety values.

*Note* This method currently only updates the conserved moiety values, it does
not update the initial species condition from which the values were calculated.

If the index vector is not given, then the values vector treated as a vector of all
values to set. If index is given, then  values should have the same
length as index.

:param numpy.ndarray index: (optional) an index array indicating which items to set,
                            or if no index array is given, the first param should be an
                            array of all the  values to set.
:param numpy.ndarray values: the values to set.


Misc
----
";



%feature("docstring") rr::ExecutableModel::getInfo "
ExecutableModel.getInfo()

get various info about the model.
";



%feature("docstring") rr::ExecutableModel::getModelName "
ExecutableModel.getModelName()

Get the model name specified in the SBML.
";



%feature("docstring") rr::ExecutableModel::getTime "
ExecutableModel.getTime()

Get the model time. The model originally start at time t=0 and is advaced forward in time by the
integrator. So, if one ran a simulation from time = 0 to time = 10, the model will then have it's
time = 10.
";



%feature("docstring") rr::ExecutableModel::reset "
ExecutableModel.reset()

Reset the floating species concentration to their initial conditions.
";



%feature("docstring") rr::ExecutableModel::setTime "
ExecutableModel.setTime(time)

Set the model time variable.

:param time: time the time value to set.
";



%feature("docstring") rr::RoadRunner "
RoadRunner.__init__(uriOrSBML = "", options = None)

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
";



%feature("docstring") rr::RoadRunner::load "
RoadRunner.load(uriOrDocument)

Loads an SBML document, given a string for file path, URI, or contents.

This method also accepts HTTP URI for remote files, however this feature is currently limited
to the Mac version, plan on enabling HTTP loading of SBML documents on Windows and Linux
shortly.

Some examples of loading files on Mac or Linux::

    >>> r.load(\"myfile.xml\")                               # load a file from the current directory
    >>> r.load(\"/Users/Fred/myfile.xml\")                   # absolute path
    >>> r.load(\"http://sbml.org/example_system.xml\")       # remote file


Or on Windows:

    >>> r.load(\"myfile.xml\")                                  # load a file from the current directory
    >>> r.load(\"file://localhost/c:/Users/Fred/myfile.xml\")   # using a URI

One may also load the contents of a document::

    >>> myfile = open(\"myfile.xml, \"r\")
    >>> contents = file.read()
    >>> r.load(contents)

In future version, we will also support loading directly from a libSBML Document object.

:param uriOrDocument: A string which may be a local path, URI or contents of an SBML document.
:type name: str
";



%feature("docstring") rr::RoadRunner::getCompiler "
RoadRunner.getCompiler()

Return the JIT :class:`Compiler` object currently being used.
This object provides various information about the current processor and system.
";



%feature("docstring") rr::RoadRunner::getConfigurationXML "
RoadRunner.getConfigurationXML()

recurse through all of the child configurable objects that this
class ones and build an assemble all of their configuration parameters
into a single xml document which is returned as a string.

The value of this result depends on what child objects are presently loaded.
";



%feature("docstring") rr::RoadRunner::getExtendedVersionInfo "
RoadRunner.getExtendedVersionInfo()

getVersion plus info about dependent libs versions.
";



%feature("docstring") rr::RoadRunner::getInfo "
RoadRunner.getInfo()

return info about the current state of the object

:rtype: str
";



%feature("docstring") rr::RoadRunner::getInstanceCount "
RoadRunner.getInstanceCount()

Number of currently running RoadRunner instances.
";



%feature("docstring") rr::RoadRunner::getInstanceID "
RoadRunner.getInstanceID()

When there are multiple instances of RoadRunner, this is the instance id.
";



%feature("docstring") rr::RoadRunner::getIntegrator "
RoadRunner.getIntegrator()

get the integrator which is currently being used to
time evolve the system.
";



%feature("docstring") rr::RoadRunner::getAvailableIntegrators "
RoadRunner.getAvailableIntegrators()

Get a list of available integrator names.
";



%feature("docstring") rr::RoadRunner::getParamPromotedSBML "
RoadRunner.getParamPromotedSBML(*args)

Takes an SBML document or path to an SBML document and changes all of the local parameters
to be global parameters.

:param str SBML: the contents or path to an SBML document
:rtype: str
";



%feature("docstring") rr::RoadRunner::getCurrentSBML "
RoadRunner.getCurrentSBML()

Returns the current state of the model in the form of an SBML string.

That is the SBML will reflect the current state of the model and not the
original SBML that was loaded into RoadRunner.

:rtype: str
";



%feature("docstring") rr::RoadRunner::getSBML "
RoadRunner.getSBML()

Returns the original SBML model that was loaded into roadrunner.

:rtype: str

Selections
----------
";



%feature("docstring") rr::RoadRunner::getIds "
RoadRunner.getIds()

Return a list of selection ids that this object can select on.

:rtype: list
";



%feature("docstring") rr::RoadRunner::getValue "
RoadRunner.getValue(sel)

Returns the value for a given selection.

:param sel: a selection that is either a string or a SelectionRecord that was
            obtained from createSelection
:type sel: str or SelectionRecord
";



%feature("docstring") rr::RoadRunner::getSelectedValues "
RoadRunner.getSelectedValues()

returns the values selected with SimulateOptions for the current model time / timestep

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::selections "

Get or set the list of current selections used for the simulation result columns.
";



%feature("docstring") rr::RoadRunner::createSelection "
RoadRunner.createSelection(sel)

Create a new selection based on a selection string

:rtype: roadrunner.SelectionRecord


Model Access
------------
";



%feature("docstring") rr::RoadRunner::isModelLoaded "
RoadRunner.isModelLoaded()

Return True if model was loaded; False otherwise
";



%feature("docstring") rr::RoadRunner::model "
:annotation: None

Get the currently loaded model. The model object contains the entire state of the SBML model.
";



%feature("docstring") rr::RoadRunner::getModel "
RoadRunner.getModel()

Function form of the RoadRunner.model property, identical to model.
";



%feature("docstring") rr::RoadRunner::oneStep "
RoadRunner.oneStep(startTime, stepSize)

Carry out a one step integration of the model. The method takes two arguments,
the current time and the step size to us in the integration. The method returns
the new time which will be currentTime + StepSize::

    newTime = rr.oneStep (10, 0.5)
";



%feature("docstring") rr::RoadRunner::reset "
RoadRunner.reset()

This method resets all the floating species concentrations to their initial values.
";



%feature("docstring") rr::RoadRunner::setConfigurationXML "
RoadRunner.setConfigurationXML(*args)

given a xml document, which should have been returned from getConfigurationXML,
this method recurses though all the child configurable elements and sets their
configuration to the values specified in the document.

:param str xml: the contents of an xml document.
";



%feature("docstring") rr::RoadRunner::conservedMoietyAnalysis "

Enables / Disables conserved moiety analysis (boolean).

If this is enabled, the SBML document (either current, or one about to be loaded)
is converted using the ConservedMoietyConverter. All of the linearly dependent
species are replaced with assignment rules and a new set of conserved moiety
parameters are introduced.


Simulation
----------

Fast and easy time series simulations is one of the main objectives of the RoadRunner project.

All simulation related tasks can be accomplished with the single ``simulate`` method.
";



%feature("docstring") rr::RoadRunner::steadyStateSelections "

A list of SelectionRecords which determine what values are used for
a steady state calculation. This list may be set by assigning a list
of valid selection symbols::

  r.steadyStateSelections = ['S1', '[S2]', 'P1']
";



%feature("docstring") rr::RoadRunner::steadyState "
RoadRunner.steadyState()

Attempt to evaluate the steady state for the model. The method returns
a value that indicates how close the solution is to the steady state.
The smaller the value the better. Values less than 1E-6 usually indicate a
steady state has been found. If necessary the method can be called a
second time to improve the solution.

:returns: the sum of squares of the steady state solution.

:rtype: double
";



%feature("docstring") rr::RoadRunner::steadyStateApproximate "
RoadRunner.steadyStateApproximate()

Attempt to approximate the steady state for the model by running integration
for a specified amount of time. This function is ideal for the case where
steady state solution does exist but the solver cannot find the 
solution due to singular Jacobian, etc. The method returns
a value that indicates how close the solution is to the steady state.
The smaller the value the better. Values less than 1E-6 usually indicate a
steady state has been found. If necessary the method can be called a
second time to improve the solution.

:returns: the sum of squares of the approximated steady state solution.

:rtype: double
";



%feature("docstring") rr::RoadRunner::getSteadyStateValues "
RoadRunner.getSteadyStateValues()

Performs a steady state calculation (evolves the system to a steady
state), then calculates and returns the set of values specified by
the steady state selections. The variable steadyStateSelections is used
to determine which values are returned.

     >>> rr.steadyStateSelections = ['S1']
     >>> rr.getSteadyStateValues()
     array([ 0.54314239])

:returns: a numpy array corresponding to the values specified by steadyStateSelections

:rtype: numpy.ndarray



Metabolic control analysis
--------------------------

In the special case when an SBML model is a purely reaction kinetics model -- no rate rules, no
assignment rules for chemical species, and time invariant stoichiometry, specialized analysis methods
related to metabolic control analysis are applicable. These methods are described in this section.
";



%feature("docstring") rr::RoadRunner::getCC "
RoadRunner.getCC(variable, parameter)

Returns a scaled control coefficient with respect to a global parameter.

For example::

  rr.getCC ('J1', 'Vmax')
  rr.getCC ('S1', 'Xo')
  rr.getCC ('S2', 'Km')

The first returns a flux control coefficient with respect to flux J1. The second and third
return concentration control coefficients with respect to species S1 and S2.

:param variable: The id of a dependent variable of the coefficient, for example a
                 flux or species concentration.

:param parameter: The id of the independent parameter, for example a kinetic constant
                  or boundary species

:returns: the value of the control coefficient returned to the caller.

:rtype: double
";


%feature("docstring") rr::RoadRunner::getuCC "
RoadRunner.getuCC(variableId, parameterId)

Get unscaled control coefficient with respect to a global parameter.

:param variableId: must be either a reaction or floating species.

:param parameterId: must be either a global parameter, boundary species, or
                    conserved sum.
";



%feature("docstring") rr::RoadRunner::getDiffStepSize "
RoadRunner.getDiffStepSize()

Returns the differential step size used in getCC and getuCC. Both functions
use a 4th order finite difference method for calculating the derivative. The
default value is 0.05.

  rr.getDiffStepSize ()
  0.05

:rtype: double
";



%feature("docstring") rr::RoadRunner::setDiffStepSize "
RoadRunner.setDiffStepSize(value)

Sets the differential step size used in getCC and getuCC.

  rr.setDiffStepSize (0.05)
";




%feature("docstring") rr::RoadRunner::getEE "
RoadRunner.getEE(reactionId, parameterId, steadyState=True)

Retrieve a single elasticity coefficient with respect to a global parameter.

For example::

  x = rr.getEE ('J1', 'Vmax')

:param str variable: The dependent variable of the coefficient, for example a flux or
                     species concentration.
:param str parameter: The independent parameter, for example a kinetic constant or boundary
                      species
:param Boolean steadyState: should the steady state value be computed.
";



%feature("docstring") rr::RoadRunner::getuEE "
RoadRunner.getuEE(reactionId, parameterId)

Get unscaled elasticity coefficient with respect to a global parameter or species.
";



%feature("docstring") rr::RoadRunner::getEigenValueIds "
RoadRunner.getEigenValueIds()

returns a list of selection symbols for the eigenvalues of the floating species. The eigen value
selection symbol is ``eigen(XX)``, where ``XX`` is the floating species name.
";



%feature("docstring") rr::RoadRunner::getFullEigenValues "
RoadRunner.getFullEigenValues()


Calculates the eigen values of the Full Jacobian as a real matrix, first column real part, second
column imaginary part.

Note, only valid for pure reaction kinetics models (no rate rules, no floating species rules and
time invariant stoichiometry).

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getReducedEigenValues "
RoadRunner.getReducedEigenValues()


Calculates the eigen values of the Reduced Jacobian as a real matrix, first column real part, second
column imaginary part.

Only valid if moiety conversion is enabled.

Note, only valid for pure reaction kinetics models (no rate rules, no floating species rules and
time invariant stoichiometry).

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getFullJacobian "
RoadRunner.getFullJacobian()

Compute the full Jacobian at the current operating point.

This is the Jacobian of ONLY the floating species.
";



%feature("docstring") rr::RoadRunner::getReducedJacobian "
RoadRunner.getReducedJacobian()

Returns the *reduced* Jacobian for the independent species. This matrix will be non-singular
for models that include moiety-conserved cycles.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getScaledConcentrationControlCoefficientMatrix "
RoadRunner.getScaledConcentrationControlCoefficientMatrix()

Returns the m by n matrix of scaled concentration control coefficients where m is the number
of floating species and n the number of reactions.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getScaledFloatingSpeciesElasticity "
RoadRunner.getScaledFloatingSpeciesElasticity(reactionId, speciesId)

Returns the scaled elasticity for a given reaction and given species.

:param str reactionId: the SBML id of a reaction.
:param str speciesId: the SBML id of a species.
:rtype: double
";



%feature("docstring") rr::RoadRunner::getUnscaledParameterElasticity "
RoadRunner.getUnscaledParameterElasticity(reactionId, parameterId)

 Returns the unscaled elasticity for a named reaction with respect to a
 named parameter

:param str reactionId: the SBML id of a reaction.
:param str parameterId: the SBML id of a parameter.
:rtype: double
";



%feature("docstring") rr::RoadRunner::getUnscaledConcentrationControlCoefficientMatrix "
RoadRunner.getUnscaledConcentrationControlCoefficientMatrix()

Returns the unscaled concentration control coefficient matrix.
";



%feature("docstring") rr::RoadRunner::getUnscaledElasticityMatrix "
RoadRunner.getUnscaledElasticityMatrix()

Returns the unscaled species elasticity matrix at the current operating point.
";



%feature("docstring") rr::RoadRunner::getUnscaledFluxControlCoefficientMatrix "
RoadRunner.getUnscaledFluxControlCoefficientMatrix()

Returns the unscaled flux control coefficient matrix.
";



%feature("docstring") rr::RoadRunner::getUnscaledSpeciesElasticity "
RoadRunner.getUnscaledSpeciesElasticity(reactionIndx, speciesIndx)

Get a single species elasticity value.


:param int reactionIndx: index of reaction
:param int speciesIndx: index of species.
";



%feature("docstring") rr::RoadRunner::getScaledFluxControlCoefficientMatrix "
RoadRunner.getScaledFluxControlCoefficientMatrix()

Returns the n by n matrix of scaled flux control coefficients where n is the number of reactions.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getScaledElasticityMatrix "
RoadRunner.getScaledElasticityMatrix()

Returns the scaled elasticity matrix at the current operating point.

:rtype: numpy.ndarray


Stochiometric Analysis
----------------------
";



%feature("docstring") rr::RoadRunner::getFullStoichiometryMatrix "
RoadRunner.getFullStoichiometryMatrix()


Get the stoichiometry matrix that coresponds to the full model, even it
it was converted via conservation conversion.
";



%feature("docstring") rr::RoadRunner::getReducedStoichiometryMatrix "
RoadRunner.getReducedStoichiometryMatrix()

get the reduced stochiometry matrix. If conservation conversion is enabled,
this is the matrix that coresponds to the independent species.

A synonym for getNrMatrix().
";



%feature("docstring") rr::RoadRunner::getConservationMatrix "
RoadRunner.getConservationMatrix()

Returns a conservation matrix :math:`\\\\Gamma` which is a :math:`c \\\\times m` matrix
where :math:`c` is the number of conservation laws and :math:`m` the number of species.
";



%feature("docstring") rr::RoadRunner::getL0Matrix "
RoadRunner.getL0Matrix()

Returns the L0 matrix for the current model. The L0 matrix is an (m-r) by r matrix that expresses
the dependent reaction rates in terms of the independent rates. m is the number of floating species
and r is the rank of the stoichiometry matrix.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getLinkMatrix "
RoadRunner.getLinkMatrix()

Returns the full link matrix, L for the current model. The Link matrix is an m by r matrix where m
is the number of floating species and r the rank of the stoichiometric matrix, N.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getNrMatrix "
RoadRunner.getNrMatrix()

Returns the reduced stoichiometry matrix, :math:`N_R`, which will have only r rows where r is the rank of
the full stoichiometry matrix. The matrix will be reordered such that the rows of :math:`N_R` are independent.

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getKMatrix "
RoadRunner.getKMatrix()

Returns the K matrix, :math:`ker(N_R)`, (right nullspace of Nr)
The K matrix has the structure, :math:`[I K0]'`

:rtype: numpy.ndarray


Analysis
--------
";



%feature("docstring") rr::RoadRunner::getFrequencyResponse "
RoadRunner.getFrequencyResponse(startFrequency, numberOfDecades, numberOfPoints, parameterName, variableName, useDB, useHz)

Compute the frequency response

:rtype: numpy.ndarray
";



%feature("docstring") rr::RoadRunner::getSupportedIdTypes "
RoadRunner.getSupportedIdTypes()

Returns a bitfield representing the element ID types which can be used with e.g.
RoadRunner.getValue

:rtype: numpy.ndarray
";



%feature("docstring") rr::SelectionRecord "
SelectionRecord.__init__(str)
Create a new selection record. This constructor really should not be called,
SelectionRecords should be created by the RoadRunner.createSelection
";



%feature("docstring") rr::SelectionRecord::TIME "

";



%feature("docstring") rr::SelectionRecord::CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::AMOUNT "

species must have either a CONCENTRATION or AMOUNT
modifer to distinguish it.
";



%feature("docstring") rr::SelectionRecord::RATE "

";



%feature("docstring") rr::SelectionRecord::BOUNDARY "

";



%feature("docstring") rr::SelectionRecord::FLOATING "

";



%feature("docstring") rr::SelectionRecord::REACTION "

";



%feature("docstring") rr::SelectionRecord::INITIAL "

";



%feature("docstring") rr::SelectionRecord::CURRENT "

";



%feature("docstring") rr::SelectionRecord::UNSCALED "

";



%feature("docstring") rr::SelectionRecord::ELASTICITY "

";



%feature("docstring") rr::SelectionRecord::CONTROL "

";



%feature("docstring") rr::SelectionRecord::EIGENVALUE "

";



%feature("docstring") rr::SelectionRecord::ELEMENT "

";



%feature("docstring") rr::SelectionRecord::STOICHIOMETRY "

";



%feature("docstring") rr::SelectionRecord::UNKNOWN "

";



%feature("docstring") rr::SelectionRecord::DEPENDENT "

";



%feature("docstring") rr::SelectionRecord::INDEPENDENT "

";



%feature("docstring") rr::SelectionRecord::CONSREVED_MOIETY "

";



%feature("docstring") rr::SelectionRecord::UNKNOWN_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::COMPARTMENT "

";



%feature("docstring") rr::SelectionRecord::GLOBAL_PARAMETER "

";



%feature("docstring") rr::SelectionRecord::FLOATING_AMOUNT "

";



%feature("docstring") rr::SelectionRecord::BOUNDARY_AMOUNT "

";



%feature("docstring") rr::SelectionRecord::BOUNDARY_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::FLOATING_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::FLOATING_AMOUNT_RATE "

";



%feature("docstring") rr::SelectionRecord::FLOATING_CONCENTRATION_RATE "

";



%feature("docstring") rr::SelectionRecord::REACTION_RATE "

";



%feature("docstring") rr::SelectionRecord::INITIAL_FLOATING_AMOUNT "

";



%feature("docstring") rr::SelectionRecord::INITIAL_FLOATING_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::UNSCALED_ELASTICITY "

";



%feature("docstring") rr::SelectionRecord::UNSCALED_CONTROL "

";



%feature("docstring") rr::SelectionRecord::UNKNOWN_ELEMENT "

";



%feature("docstring") rr::SelectionRecord::ALL "

";



%feature("docstring") rr::SelectionRecord::ALL_INDEPENDENT "

";



%feature("docstring") rr::SelectionRecord::ALL_DEPENDENT "

";



%feature("docstring") rr::SelectionRecord::ALL_INDEPENDENT_AMOUNT "

";



%feature("docstring") rr::SelectionRecord::ALL_DEPENDENT_AMOUNT "

";



%feature("docstring") rr::SelectionRecord::ALL_INDEPENDENT_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::ALL_DEPENDENT_CONCENTRATION "

";



%feature("docstring") rr::SelectionRecord::index "
:annotation: int
";



%feature("docstring") rr::SelectionRecord::p1 "
:annotation: str
";



%feature("docstring") rr::SelectionRecord::p2 "
:annotation: str
";



%feature("docstring") rr::SimulateOptions::concentrations "


A list of the variable whose output in the results file is in concentration
(not amount) units. This list of variables must be a subset of the names
listed in variables.
";



%feature("docstring") rr::SimulateOptions::duration "


The duration of the simulation run, in the model's units of time. Note,
setting the duration automatically sets the end time and visa versa.
";



%feature("docstring") rr::SimulateOptions::end "


The simulation end time. Note, setting the end time automatically sets the
duration accordingly and visa versa.
";



%feature("docstring") rr::SimulateOptions::flags "


can be set to ResetModel so that the model is reset to its initial state
when the simulation is run.
";



%feature("docstring") rr::SimulateOptions::variables "


The variables (in addition to time) whose values will be saved in the result.
These are SBML model id's. Order is significant, as this determines the order
of the columns in the result matrix.

Important: if a symbol in this list refers to a species in the model,
then that symbol should also be listed in either the amount or concentration
lists below. If a species symbol is listed in variables, but is not listed
in either amounts or concentrations, then it defaults to an amount value.

The ordering of the symbols in variable is what determines the output
ordering. The order of symbols in either amounts or concentrations do not
effect the output ordering.

NOTE:If a listed variable has two underscores in it ('__'), that variable
is actually present only in a submodel of the main model, from the
Hierarchical Model Composition package, in the format submodelID__variableID.
If the model is flattened, the variable will appear automatically.
";



%feature("docstring") rr::LoadSBMLOptions::conservedMoieties "
:annotation: bool

perform conservation analysis.

This causes a re-ordering of the species, so results generated
with this flag enabled can not be compared index wise to results
generated otherwise.

Moiety conservation is only compatable with simple models which do NOT have any events or rules
which define or alter any floating species, and which have simple constant stiochiometries.
";



%feature("docstring") rr::LoadSBMLOptions::mutableInitialConditions "
:annotation: bool

Generate accessor functions to allow changing of initial
conditions.
";



%feature("docstring") rr::LoadSBMLOptions::noDefaultSelections "
:annotation: bool

Do not create a default selection list when the model is loaded.
";



%feature("docstring") rr::LoadSBMLOptions::readOnly "
:annotation: bool

Should the model be recompiled?
The LLVM ModelGenerator maintins a hash table of currently running
models. If this flag is NOT set, then the generator will look to see
if there is already a running instance of the given model and
use the generated code from that one.

If only a single instance of a model is run, there is no
need to cache the models, and this can safetly be enabled,
realizing some performance gains.
";



%feature("docstring") rr::LoadSBMLOptions::recompile "
:annotation: bool

If this is set, then a read-only model is generated. A read-only
model can be simulated, but no code is generated to set model
values, i.e. parameters, amounts, values, etc...

It takes a finite amount of time to generate the model value setting
functions, and if they are not needed, one may see some performance
gains, especially in very large models.
";



%feature("docstring") rr::Logger::LOG_CURRENT "

Use the current level -- don't change the level from what it is.
";



%feature("docstring") rr::Logger::LOG_FATAL "

A fatal error. The application will most likely terminate. This is the highest priority.
";



%feature("docstring") rr::Logger::LOG_CRITICAL "

A critical error. The application might not be able to continue running successfully.
";



%feature("docstring") rr::Logger::LOG_ERROR "

An error. An operation did not complete successfully, but the application as a whole is not affected.
";



%feature("docstring") rr::Logger::LOG_WARNING "

A warning. An operation completed with an unexpected result.
";



%feature("docstring") rr::Logger::LOG_NOTICE "

A notice, which is an information with just a higher priority.
";



%feature("docstring") rr::Logger::LOG_INFORMATION "

An informational message, usually denoting the successful completion of an operation.
";



%feature("docstring") rr::Logger::LOG_DEBUG "

A debugging message.
";



%feature("docstring") rr::Logger::LOG_TRACE "

A tracing message. This is the lowest priority.

Logging Methods
---------------
";



%feature("docstring") rr::Logger::setLevel "
Logger.setLevel([level])

sets the logging level to one a value from Logger::Level

:param int level: the level to set, defaults to LOG_CURRENT if none is specified.
";



%feature("docstring") rr::Logger::getLevel "
Logger.getLevel()

get the current logging level.
";



%feature("docstring") rr::Logger::disableLogging "
Logger.disableLogging()

Suppresses all logging output
";



%feature("docstring") rr::Logger::disableConsoleLogging "
Logger.disableConsoleLogging()

stops logging to the console, but file logging may continue.
";



%feature("docstring") rr::Logger::enableConsoleLogging "
Logger.enableConsoleLogging(level)

turns on console logging (stderr) at the given level.

:param level: A logging level, one of the above listed LOG_* levels.
";



%feature("docstring") rr::Logger::enableFileLogging "
Logger.enableFileLogging(fileName, [level])

turns on file logging to the given file as the given level.

:param str fileName: the path of a file to log to.
:param level: (optional) the logging level, defaults to LOG_CURRENT.
";



%feature("docstring") rr::Logger::disableFileLogging "
Logger.disableFileLogging()

turns off file logging, but has no effect on console logging.
";



%feature("docstring") rr::Logger::getCurrentLevelAsString "
Logger.getCurrentLevelAsString()

get the textural form of the current logging level.
";



%feature("docstring") rr::Logger::getFileName "
Logger.getFileName()

get the name of the currently used log file.
";



%feature("docstring") rr::Logger::setFormattingPattern "
Logger.setFormattingPattern(format)

Internally, RoadRunner uses the Poco logging framework, so we
can custom format logging output based on a formatting pattern
string.

The format pattern is used as a template to format the message and
is copied character by character except for the following special characters,
which are replaced by the corresponding value.

An example pattern of \"%Y-%m-%d %H:%M:%S %p: %t\" set via::

  roadrunner.Logger.setFormattingPattern(\"%Y-%m-%d %H:%M:%S %p: %t\")

would produce the following output:

.. image:: logging2.png

|

RoadRunner supports the following format specifiers. These were copied from the Poco documentation:

* %s - message source
* %t - message text
* %l - message priority level (1 .. 7)
* %p - message priority (Fatal, Critical, Error, Warning, Notice, Information, Debug, Trace)
* %q - abbreviated message priority (F, C, E, W, N, I, D, T)
* %P - message process identifier
* %T - message thread name
* %I - message thread identifier (numeric)
* %N - node or host name
* %U - message source file path (empty string if not set)
* %u - message source line number (0 if not set)
* %w - message date/time abbreviated weekday (Mon, Tue, ...)
* %W - message date/time full weekday (Monday, Tuesday, ...)
* %b - message date/time abbreviated month (Jan, Feb, ...)
* %B - message date/time full month (January, February, ...)
* %d - message date/time zero-padded day of month (01 .. 31)
* %e - message date/time day of month (1 .. 31)
* %f - message date/time space-padded day of month ( 1 .. 31)
* %m - message date/time zero-padded month (01 .. 12)
* %n - message date/time month (1 .. 12)
* %o - message date/time space-padded month ( 1 .. 12)
* %y - message date/time year without century (70)
* %Y - message date/time year with century (1970)
* %H - message date/time hour (00 .. 23)
* %h - message date/time hour (00 .. 12)
* %a - message date/time am/pm
* %A - message date/time AM/PM
* %M - message date/time minute (00 .. 59)
* %S - message date/time second (00 .. 59)
* %i - message date/time millisecond (000 .. 999)
* %c - message date/time centisecond (0 .. 9)
* %F - message date/time fractional seconds/microseconds (000000 - 999999)
* %z - time zone differential in ISO 8601 format (Z or +NN.NN)
* %Z - time zone differential in RFC format (GMT or +NNNN)
* %E - epoch time (UTC, seconds since midnight, January 1, 1970)
* %[name] - the value of the message parameter with the given name
* %% - percent sign

:param str format: the logging format string. Must be formatted using the above specifiers.
";



%feature("docstring") rr::Logger::getFormattingPattern "
Logger.getFormattingPattern()

get the currently set formatting pattern.
";



%feature("docstring") rr::Logger::levelToString "
Logger.levelToString(level)

gets the textual form of a logging level Enum for a given value.

:param int level: One of the above listed logging levels.
";



%feature("docstring") rr::Logger::stringToLevel "
Logger.stringToLevel(s)

parses a string and returns a Logger::Level

:param str s: the string to parse.
";



%feature("docstring") rr::Logger::getColoredOutput "
Logger.getColoredOutput()

check if we have colored logging enabled.
";



%feature("docstring") rr::Logger::setColoredOutput "
Logger.setColoredOutput(b)

enable / disable colored output

:param boolean b: turn colored logging on or off
";



%feature("docstring") rr::Logger::setProperty "
Logger.setProperty(name, value)

Set the color of the output logging messages.

In the future, we may add additional properties here.

The following properties are supported:

* enableColors:      Enable or disable colors.
* traceColor:        Specify color for trace messages.
* debugColor:        Specify color for debug messages.
* informationColor:  Specify color for information messages.
* noticeColor:       Specify color for notice messages.
* warningColor:      Specify color for warning messages.
* errorColor:        Specify color for error messages.
* criticalColor:     Specify color for critical messages.
* fatalColor:        Specify color for fatal messages.


The following color values are supported:

* default
* black
* red
* green
* brown
* blue
* magenta
* cyan
* gray
* darkgray
* lightRed
* lightGreen
* yellow
* lightBlue
* lightMagenta
* lightCyan
* white

:param str name: the name of the value to set.
:param str value: the value to set.
";



%feature("docstring") rr::Logger::log "
Logger.log(level, msg)

logs a message to the log.

:param int level: the level to log at.
:param str msg: the message to log.
";



%feature("docstring") rr::Config::setValue "
Config.setValue(key, value)

Set the value of a configuration key. The value must be either a string, integer, double or
boolean. If one wanted to change the value of the default integrator tolerances, one would::

  from roadrunner import Config
  Config.setValue(Config.SIMULATEOPTIONS_ABSOLUTE, 3.14e-12)
  Config.setValue(Config.SIMULATEOPTIONS_RELATIVE, 2.78e-5)


Or, other options may be set to Boolean or integer values. To enable an optimization features,
or to set default simulation time steps::

  Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
  Config.setValue(Config.SIMULATEOPTIONS_STEPS, 100)
";



%feature("docstring") rr::Config::getConfigFilePath "
Config.getConfigFilePath()

If roadrunner was able to find a configuration file on the file system, its full path is returned
here. If no file was found, this returns a empty string.
";



%feature("docstring") rr::Config::readConfigFile "
Config.readConfigFile(path)

Read all of the values from a configuration file at the given path. This overrides any previously
stored configuration. This allows users to have any number of configuration files and load them
at any time. Say someone had to use Windows, and they had a file in thier C: drive, this would be
loaded via::

  Config.readConfigFile(\"C:/my_config_file.txt\")

Note, the forward slash works on both Unix and Windows, using the forward slash eliminates the
need to use a double back slash, \"\\\\\\\\\\\\\\\\\".
";



%feature("docstring") rr::Config::writeConfigFile "
Config.writeConfigFile(path)

Write all of the current configuration values to a file. This could be written to one of the
default locations, or to any other location, and re-loaded at a later time.



Available Configuration Parameters
----------------------------------
All of the configuration parameter keys are static attributes of the Config class and are listed
here. The variable type of the parameter is listed after the key name.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES "
:annotation: bool

perform conservation analysis.

This causes a re-ordering of the species, so results generated
with this flag enabled can not be compared index wise to results
generated otherwise.

Moiety conservation is only compatable with simple models which do NOT have any events or rules
which define or alter any floating species, and which have simple constant stoichiometries.

Moiety conservation may cause unexpected results, be aware of what it is before enableing.

Not recommended for time series simulations.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_RECOMPILE "
:annotation: bool

Should the model be recompiled?
The LLVM ModelGenerator maintains a hash table of currently running
models. If this flag is NOT set, then the generator will look to see
if there is already a running instance of the given model and
use the generated code from that one.

If only a single instance of a model is run, there is no
need to cache the models, and this can safely be enabled,
realizing some performance gains.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_READ_ONLY "
:annotation: bool

If this is set, then a read-only model is generated. A read-only
model can be simulated, but no code is generated to set model
values, i.e. parameters, amounts, values, etc...

It takes a finite amount of time to generate the model value setting
functions, and if they are not needed, one may see some performance
gains, especially in very large models.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS "
:annotation: bool

Generate accessors functions to allow changing of initial
conditions.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_GVN "
:annotation: bool

GVN - This pass performs global value numbering and redundant load
elimination cotemporaneously.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION "
:annotation: bool

CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
simplify terminator instructions, etc...
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING "
:annotation: bool

InstructionCombining - Combine instructions to form fewer, simple
instructions. This pass does not modify the CFG, and has a tendency to make
instructions dead, so a subsequent DCE pass is useful.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION "
:annotation: bool

DeadInstElimination - This pass quickly removes trivially dead instructions
without modifying the CFG of the function.  It is a BasicBlockPass, so it
runs efficiently when queued next to other BasicBlockPass's.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION "
:annotation: bool

DeadCodeElimination - This pass is more powerful than DeadInstElimination,
because it is worklist driven that can potentially revisit instructions when
their other instructions become dead, to eliminate chains of dead
computations.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER "
:annotation: bool


InstructionSimplifier - Remove redundant instructions.
";



%feature("docstring") rr::Config::LOADSBMLOPTIONS_USE_MCJIT "
:annotation: bool

Currently disabled.

Use the LLVM MCJIT JIT engine.

Defaults to false.

The MCJIT is the new LLVM JIT engine, it is not as well tested as the
original JIT engine. Does NOT work on LLVM 3.1
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_STEPS "
:annotation: int

The number of steps at which the output is sampled. The samples are evenly spaced.
When a simulation system calculates the data points to record, it will typically
divide the duration by the number of time steps. Thus, for N steps, the output
will have N+1 data rows.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_DURATION "
:annotation: double

The duration of the simulation run, in the model's units of time. Note,
setting the duration automatically sets the end time and visa versa.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_ABSOLUTE "
:annotation: double

The absolute error tolerance used by the integrator.

A number representing the absolute difference permitted for the integrator
tolerance. Defaults to 1.000000e-10.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_RELATIVE "
:annotation: double

The relative error tolerance used by the integrator.

A float-point number representing the relative difference permitted.
Defaults to 1.000000e-05.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_STRUCTURED_RESULT "
:annotation: bool

A boolean option to return a structured array from the RoadRunner.simulate method. Structured
arrays contain column names and other data. A structured array needs to be converted into regular
arrays before they can be used in numpy functions.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_STIFF "
:annotation: bool

Is the model a stiff system? setting this to stiff causes
RoadRunner to load a stiff solver which could potentially be
extremly slow
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_MULTI_STEP "
:annotation: bool

The MULTI_STEP option tells the solver to take a series of internal steps
and then return the solution at the point reached by that step.

In simulate, this option will likely be slower than normal mode,
but may be useful to monitor solutions as they are integrated.

This is intended to be used in combination with the
IntegratorListener. It this option is set, and there is a
IntegratorListener set, RoadRunner::integrate will run the
integrator in a series of internal steps, and the listner
will by notified at each step.

Highly Experimental!!!
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_INITIAL_TIMESTEP "
:annotation: double

A user specified initial time step. If this is <=  0, the integrator
will attempt to determine a safe initial time step.

Note, for each number of steps given to RoadRunner::simulate or RoadRunner::oneStep,
the internal integrator may take many many steps to reach one of the external time
steps. This value specifies an initial value for the internal integrator
time step.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP "
:annotation: double

Specify The Minimum Time Step That The Internal Integrator
Will Use. Uses Integrator Estimated Value If <= 0.
";



%feature("docstring") rr::Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS "
:annotation: int

Specify The Maximum Number Of Steps The Internal Integrator Will Use
Before Reaching The User Specified Time Span. Uses The Integrator
Default Value If <= 0.
";



%feature("docstring") rr::Config::ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES "
:annotation: int

RoadRunner by default dynamically generates accessors properties
for all SBML symbol names on the model object when it is retrieved
in Python. This feature is very nice for interactive use, but
can slow things down. If this feature is not needed, it
can be disabled here.
";



%feature("docstring") rr::Config::ROADRUNNER_DISABLE_WARNINGS "
:annotation: int

disable SBML conserved moiety warnings.

Conserved Moiety Conversion may cause unexpected behavior, be aware of what it
is before enabling.

RoadRunner will issue a warning in steadyState if conservedMoieties are NOT
enabled because of a potential singular Jacobian. To disable this warning,
set this value to 1

A notice will be issued whenever a document is loaded and conserved moieties
are enabled. To disable this notice, set this value to 2.

To disable both the warning and notice, set this value to 3

Rationale for these numbers: This is actual a bit field, disabling the steady state
warning value is actually 0b01 << 0 which is 1, and the loading warning is 0b01 << 1
which is 2 and 0b01 & 0b10 is 0b11 which is 3 in decimal.
";



%feature("docstring") rr::PyConservedMoietyConverter::setDocument "
PyConservedMoietyConverter.setDocument(sbmlOrURI)

Sets the source document. This may be either the contents of an sbml docment
as a string, or the path the the file.

:param str sbmlOrURI: contents or path of source document
:returns: an integer indicating success or failure, 0 means success.
:rtype: int
";



%feature("docstring") rr::PyConservedMoietyConverter::convert "
PyConservedMoietyConverter.convert()

Perform the document conversion.

:returns: an integer indicating success or failure, 0 means success.
:rtype: int
";



%feature("docstring") rr::PyConservedMoietyConverter::getDocument "
PyConservedMoietyConverter.getDocument()

Get the converted document contents.

:returns: The contents of the converted document, or empty string
          if there is no source document.
";




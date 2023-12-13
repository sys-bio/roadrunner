Changing Initial Conditions
===========================

There are a number of methods to get and set the initial conditions of a loaded model. In order to
specify a given initial conditions we use the notation, ``init()``.  The values stored in the
initial conditions are applied to the model whenever it is reset. The list of all initial condition
symbols can be obtained by the methods, :meth:`~ExecutableModel.getFloatingSpeciesInitAmountIds()`
and :meth:`~ExecutableModel.getFloatingSpeciesInitConcentrationIds()` assuming ``rr`` is a RoadRunner
instance. As with all other selection symbols, the :meth:`~ExecutableModel.keys()` returns all
available selection symbols:

  >>>  rr.model.keys()
  [ 'S1', 'S2', '[S1]', '[S2]', 'compartment', 'k1', '_CSUM0',
    'reaction1',  'init([S1])',  'init([S2])', 'init(S1)',
    'init(S2)',  "S1'"]

Symbols for selecting initial values specifically for amounts and concentrations can be obtained
via:

  >>> rr.model.getFloatingSpeciesInitAmountIds()
  ['init(S1)', 'init(S2)']

  >>> rr.model.getFloatingSpeciesInitConcentrationIds()
  ['init([S1])', 'init([S2])']

Getting or setting initial values is easily accomplished using the array operator and the selection
symbols:

  >>> rr.model["init(S1)"]
  0.00015

  >>> rr.model["init([S1])"]
  2.9999999999999997e-05

  >>> rr.model["init([S1])"] = 2

  >>> rr.model["init(S1)"]
  10.0

The values for the initial conditions for all floating species can be obtained using the calls:

  >>> rr.model.getFloatingSpeciesInitConcentrations()
  array([ 0.7,  5.6])

Initial conditions can be set using the two methods for all species in one call:

  >>> rr.model.setFloatingSpeciesInitAmounts ([3.4, 5.6])


  >>> rr.model.setFloatingSpeciesInitConcentrations ([6.7, 0.1])

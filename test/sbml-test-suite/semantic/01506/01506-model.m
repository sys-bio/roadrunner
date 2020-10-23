(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule, and affected by an event.
componentTags:   AssignmentRule, CSymbolTime, Compartment, EventNoDelay, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, VolumeConcentrationRates
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 NOTE:  This test is a more complicated version of test 1779.  If this isn't passing, make sure you pass test 1779 first, as it is likely that the issue here is actually reflective of the more fundamental issue tested in that test (i.e. the simultaneous assignment to a species concentration and compartment).
 
 A species in a non-unity compartment under control of a rate rule, and the species is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by the rate of change.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.  Partway through the simulation, an event changes the level of the species and of the compartment, forcing yet another change.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time >= 0.45$ | $S1 = 0.2$ |
|  |  | $C1 = 0.2$ |]


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0.4$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

*)

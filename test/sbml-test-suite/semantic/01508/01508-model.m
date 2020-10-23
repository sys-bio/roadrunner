(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule, then modified by an event.
componentTags:   AssignmentRule, CSymbolTime, Compartment, EventWithDelay, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, EventUsesTriggerTimeValues, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, VolumeConcentrationRates
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A species in a non-unity compartment under control of a rate rule, and the species is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by the rate of change.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.  The species concentration is changed by an event partway through the simulation, affecting things all over again, and that event has a delay, with an assignment dependent on the correct value for S1, calculated at trigger time.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time >= 0.45$ | $0.1$ | $S1 = S1 + 0.2$ |]


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

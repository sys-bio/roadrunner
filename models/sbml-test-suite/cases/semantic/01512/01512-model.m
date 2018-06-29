(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule,that affects an event priority.
componentTags:   AssignmentRule, CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, VolumeConcentrationRates
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A species in a non-unity compartment under control of a rate rule, and the species is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by the rate of change.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.  Two other simultaneous events fire, with the priority of one equal to S1, and the priority of the other equal to a value between the concentration of S1 and the amount of S1, to ensure that the order is correct.

The model contains:
* 1 species (S1)
* 2 parameters (x, x2)
* 1 compartment (C1)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 0.45$ | $S1$ | $x2 = 3$ |
| E1 | $time > 0.45$ | $0.15$ | $x2 = 5$ |]


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0.4$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial value of parameter x2 | $0$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

*)

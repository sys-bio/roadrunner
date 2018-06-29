(*

category:        Test
synopsis:        Two simultaneous events whose priorities involve a substance-only species.
componentTags:   CSymbolTime, Compartment, EventNoDelay, EventPriority, RateRule, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two simultaneous events have similar priorities, with one being greater than the other because of the actual value of the substance-only speces.  If misinterpreted, the two events will fire in the wrong order..

The model contains:
* 1 species (S1)
* 1 compartment (C)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time >= 2.5$ | $S1$ | $S1 = 5$ |
| E1 | $time >= 2.5$ | $7.4$ | $S1 = 10$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $-1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $10$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

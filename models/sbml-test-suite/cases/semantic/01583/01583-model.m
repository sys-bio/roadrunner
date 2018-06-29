(*

category:        Test
synopsis:        Two simultaneous events which both assign to a stoichiometry according to their priorities.
componentTags:   Compartment, EventNoDelay, EventPriority, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The two events in this model both assign to a stoichiometry at the same time, meaning that the higher-priority event fires first, and is then overridden by the lower-priority event.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $k1 > 4.5$ | $10$ | $S1_stoich = 2$ |
| E1 | $k1 > 4.5$ | $5$ | $S1_stoich = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter k1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

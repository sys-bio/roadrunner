(*

category:        Test
synopsis:        A non-persistent event that references a stoichiometry.
componentTags:   Compartment, EventNoDelay, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, EventT0Firing, NonConstantParameter, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a non-persistent event is triggered by changes in a stoichiometry, but not for long enough, and it doesn't execute.

The model contains:
* 1 species (S1)
* 2 parameters (P1, k1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E1 | $S1_stoich <= 3.3$ | false | $P1 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1_stoich | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $2$ | constant |
| Initial value of parameter P1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

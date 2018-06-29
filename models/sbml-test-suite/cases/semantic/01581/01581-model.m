(*

category:        Test
synopsis:        A persistent event whose delayed assignment affects a reaction's stoichiometry.
componentTags:   Compartment, EventWithDelay, Parameter, RateRule, Reaction, Species, StoichiometryMath
testTags:        Amount, AssignedVariableStoichiometry, EventIsPersistent, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 The event in this model is persistent, and fires even though its trigger reverts to 'false' before executing.  It assigns a value to a parameter, which in turn assigns to the stoichiometry of the reaction.  This is the same model as 1580, but uses L2's StoichiometryMath construct to assign the value.

The model contains:
* 1 species (S1)
* 2 parameters (k1, P1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $(k1 > 4.5) && (k1 < 5.1)$ | $2$ | $P1 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of parameter P1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $P1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

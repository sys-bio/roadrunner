(*

category:        Test
synopsis:        An assigned variable stoichiometry is given a delay.
componentTags:   CSymbolDelay, CSymbolTime, Compartment, EventNoDelay, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, DelayInEventAssignment, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the delayed value of P0 is assigned to the stoichiometry of the reaction J0.

The model contains:
* 1 species (S1)
* 1 parameter (P1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 3$ | $S1_stoich = delay(P1, 1)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of parameter P1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

(*

category:        Test
synopsis:        A rate of a species changed by an assigned stoichiometry
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, InitialValueReassigned, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, P0 is assigned the rate of S1, which is under control of a reaction with an assigned stoichiometry.

The model contains:
* 1 species (S1)
* 1 parameter (P0)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1 / 2$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

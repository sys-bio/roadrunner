(*

category:        Test
synopsis:        Assigned stoichiometries, both constant and variable.
componentTags:   Compartment, InitialAssignment, RateRule, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, AssignedVariableStoichiometry, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model has a reaction with both an assigned constant stoichiometry, and an assigned variable stoichiometry.

The model contains:
* 2 species (S1, S2)
* 1 compartment (C)
* 2 species references (S1_degrade, S2_create)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1_degrade S1 -> S2_create S2 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_degrade, S2_create


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S2_create | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $30$ | variable |
| Initial amount of species S2 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_degrade' | $3$ | constant |
| Initial value of species reference 'S2_create' | $1$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

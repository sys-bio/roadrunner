(*

category:        Test
synopsis:        Assigned stoichiometries, both constant and variable.
componentTags:   Compartment, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, BoundaryCondition, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model has a reaction whose sole participant is a boundary species, whose stoichiometry changes in time... which obviously does nothing at all.

The model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1_stoich S1 -> | $1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1_stoich | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

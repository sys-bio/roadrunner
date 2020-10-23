(*

category:        Test
synopsis:        A rate of a boundary species that does not change.
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the rate of S1 and S2 are assigned to P1 and P2, with S1 being a boundary species that does not change.

The model contains:
* 2 species (S1, S2)
* 2 parameters (P1, P2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $rateOf(S1)$ |
| Assignment | P2 | $rateOf(S2)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter P1 | $rateOf(S1)$ | variable |
| Initial value of parameter P2 | $rateOf(S2)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

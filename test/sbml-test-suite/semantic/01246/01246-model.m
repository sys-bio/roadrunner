(*

category:        Test
synopsis:        Two Reactions; one with no reactants or products.
componentTags:   Compartment, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that a Reaction with no reactants nor produts is treated properly when another normal Reaction is present.

The model contains:
* 1 species (S1)
* 1 compartment (c)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> | $2$ |
| J1: -> S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

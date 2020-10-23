(*

category:        Test
synopsis:        A Reaction with no reactants or products.
componentTags:   Compartment, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that a Reaction with no reactants nor produts is treated properly.

The model contains:
* 1 species (S1)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> | $2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

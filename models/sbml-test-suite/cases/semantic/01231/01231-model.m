(*

category:        Test
synopsis:        A simple reaction whose identifier is used in another reaction's kinetic law.
componentTags:   Compartment, Parameter, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to check using the identifier of a reaction (J0) in another reaction (J1).

The model contains:
* 2 species (S1, S2)
* 1 parameter (k1)
* 1 compartment (c)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |
| J1: -> S2 | $J0 + 1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial concentration of species S2 | $0$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial volume of compartment 'c' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


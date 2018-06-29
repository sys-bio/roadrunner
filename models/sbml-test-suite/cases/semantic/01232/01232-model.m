(*

category:        Test
synopsis:        A simple reaction whose identifier is shadowed by another reaction's local parameter.
componentTags:   Compartment, Parameter, Reaction, Species
testTags:        Amount, LocalParameters
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to check shadowing the identifier of a reaction (J0) with a local parameter of another reaction (J1).

The model contains:
* 2 species (S1, S2)
* 1 parameter (k1)
* 1 local parameter (J1.J0)
* 1 compartment (c)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |
| J1: -> S2 | $J0$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial concentration of species S2 | $0$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial value of local parameter 'J1.J0' | $2$ | constant |
| Initial volume of compartment 'c' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


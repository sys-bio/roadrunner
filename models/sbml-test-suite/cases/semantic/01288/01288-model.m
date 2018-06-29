(*

category:        Test
synopsis:        Use of a Boolean in kinetic laws.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, BoolNumericSwap
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

The two reactions have kinetic laws of 'true' and 'false', causing the former to have a rate of 1, and the latter a rate of 0.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> | $false$ |
| J1: -> S1 | $true$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

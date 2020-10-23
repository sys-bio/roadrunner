(*

category:        Test
synopsis:        Two fast reactions whose rates depend on initial assignments.
componentTags:   CSymbolTime, Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, FastReaction, InitialValueReassigned, MultipleFastReactions
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

The model tests fast reactions which are positive or negative depending on the results of initial assignments.

The model contains:
* 4 species (S1, S2, S3, S4)
* 2 parameters (k1, k2)
* 1 compartment (C)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> S2 | $S1 * S2 * k1$ | fast |
| J1: S3 -> S4 | $S3 * S4 * k2$ | fast |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial concentration of species S3 | $4$ | variable |
| Initial concentration of species S4 | $4$ | variable |
| Initial value of parameter k1 | $0.5 - time$ | variable |
| Initial value of parameter k2 | $time - 0.5$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

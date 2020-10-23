(*

category:        Test
synopsis:        A fast reaction with assigned stoichiometry.
componentTags:   AssignmentRule, Compartment, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, FastReaction, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

The model tests a fast reaction with assigned stoichiometry.

The model contains:
* 2 species (S1, S2)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1_stoich S1 -> S2 | $S1 * S2 * 0.1$ | fast |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1_stoich | $2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

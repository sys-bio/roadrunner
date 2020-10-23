(*

category:        Test
synopsis:        Local parameters shadowing two species references and the reaction ID of their own reaction.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, LocalParameters
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, local parameters shadow the ID of species references from the reaction they are in, as well as the ID of the reaction itself.

The model contains:
* 2 species (S1, S2)
* 3 local parameters (J1.S1_stoich, J1.S2_stoich, J1.J1)
* 1 compartment (C)
* 2 species references (S1_stoich, S2_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: S1 -> S2 | $S1_stoich * S2_stoich * J1 * 200$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of local parameter 'J1.S1_stoich' | $0.1$ | constant |
| Initial value of local parameter 'J1.S2_stoich' | $0.1$ | constant |
| Initial value of local parameter 'J1.J1' | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | constant |
| Initial value of species reference 'S2_stoich' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

(*

category:        Test
synopsis:        A local parameter shadowing a species reference from its own reaction.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, LocalParameters
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter shadows the ID of a species reference from the reaction it is in.  This is valid even in SBML L2, where you cannot reference a species reference ID anywhere... because it's not being referenced.

The model contains:
* 1 species (S1)
* 1 local parameter (J1.S1_stoich)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: -> S1 | $S1_stoich * 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of local parameter 'J1.S1_stoich' | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

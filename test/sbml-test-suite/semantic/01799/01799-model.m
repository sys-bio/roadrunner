(*

category:        Test
synopsis:        A local parameter shadowing its own reaction's ID.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, LocalParameters
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter shadows the ID of the reaction it is in.  Its kinetic law, therefore, is actually OK and not a circular reference.

The model contains:
* 1 species (S1)
* 1 local parameter (J1.J1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: -> S1 | $J1 * 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of local parameter 'J1.J1' | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

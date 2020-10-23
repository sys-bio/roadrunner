(*

category:        Test
synopsis:        Multiple species references to the same species
componentTags:   Compartment, Reaction, Species
testTags:        Amount, NonUnityStoichiometry
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this test, a single reaction has two products that both point to the same species.  This is equivalent to a reaction with a single product whose stoichiometry is equal to the sum of the stoichiometries of the doubled species references.

The model contains:
* 1 species (A)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> A + 2A | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

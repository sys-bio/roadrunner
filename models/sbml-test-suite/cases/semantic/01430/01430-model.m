(*

category:        Test
synopsis:        Multiple species references to the same species
componentTags:   Compartment, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this test, a single reaction has multiple reactants and products that both point to the same species.  This is equivalent to a reaction where all of the species references that point to the same species are collapsed into a single species reference with a stoichiometry equal to the sum of the stoichiometries of the original species references.

The model contains:
* 2 species (A, B)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: A + A -> B + B | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A | $20$ | variable |
| Initial concentration of species B | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

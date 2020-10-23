(*

category:        Test
synopsis:        Default values for species boundaryCondition
componentTags:   Compartment, Reaction, Species
testTags:        Amount, DefaultValue, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In SBML L2, the value for a speciesReference 'stoichiometry' attribute defaults to '1'.  This model tests that this value is correctly assumed, in a model where the first speciesReference has no 'stoichiometry' attribute, the second has it set to '2', and the third has it set to '1'.  The first should follow the third.

The model contains:
* 3 species (S1, S2, S3)
* 1 compartment (C)

There are 3 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $1$ |
| J1: -> 2S2 | $1$ |
| J2: -> S3 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial concentration of species S3 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

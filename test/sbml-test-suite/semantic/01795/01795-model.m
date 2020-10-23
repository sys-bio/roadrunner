(*

category:        Test
synopsis:        Default values for species boundaryCondition
componentTags:   Compartment, Reaction, Species
testTags:        Amount, BoundaryCondition, DefaultValue
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In SBML L1 and L2, the value for a species 'boundaryCondition' attribute defaults to 'false'.  This model tests that this value is correctly assumed, in a model where S1 has no 'boundaryCondition' attribute, S2 has it set to 'true', and S3 has it set to 'false'.  S1 should follow S3, and not S2.
 
The model contains:
* 3 species (S1, S2, S3)
* 1 compartment (C)

There are 3 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $1$ |
| J1: -> S2 | $1$ |
| J2: -> S3 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $2$ | variable |
| Initial concentration of species S2 | $2$ | variable |
| Initial concentration of species S3 | $2$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

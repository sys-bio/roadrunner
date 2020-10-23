(*

category:        Test
synopsis:        Default values for species boundaryCondition
componentTags:   AssignmentRule, Compartment, Parameter, Species
testTags:        Amount, DefaultValue, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In SBML L1 and L2, the value for a species 'hasOnlySubstanceUnits' attribute defaults to 'false'.  This model tests that this value is correctly assumed, in a model where S1 has no 'hasOnlySubstanceUnits' attribute, S2 has it set to 'false', and S3 has it set to 'true'.  S1 should follow S2, and not S3.

The model contains:
* 3 species (S1, S2, S3)
* 3 parameters (x, y, z)
* 1 compartment (C)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $S3$ |
| Assignment | y | $S2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $2$ | variable |
| Initial amount of species S3 | $2$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial value of parameter y | $S2$ | variable |
| Initial value of parameter z | $S3$ | variable |
| Initial volume of compartment 'C' | $3$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

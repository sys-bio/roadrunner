(*

category:        Test
synopsis:        Test of proper interpretation of a boundary species set hasOnlySubstanceUnits=true in a reaction rate.
componentTags:   Compartment, RateRule, Reaction, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, NonConstantCompartment, NonUnityCompartment
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 This model is a further simplification of test 1000/1121 that only test the use of a boundary species, used in the model in terms of concentrations, from a compartment that varies in time, used in a reaction rate, with hasOnlySubstanceUnits = true.

The model contains:
* 2 species (S1, S3)
* 1 compartment (comp)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| _J0: S1 -> | $S3 / 10$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | comp | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial amount of species S3 | $4$ | variable |
| Initial volume of compartment 'comp' | $5$ | variable |]

*)


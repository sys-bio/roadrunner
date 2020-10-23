(*

category:      Test
synopsis:      A model that tests how not to vary a species.
componentTags: Compartment, RateRule, Species
testTags:      Amount, BoundaryCondition, ConstantSpecies, NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model tests that the simulator should, by default, assume that a species set boundaryCondition=true and hasOnlySubstanceUnits=false, residing in a compartment that changes over time, should keep its *amount* constant, not its concentration.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | C | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | constant |
| Initial volume of compartment 'C' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


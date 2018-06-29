(*

category:        Test
synopsis:        A boundary substance-only species in non-unit compartment under control of a rate rule.
componentTags:   Compartment, RateRule, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

In this model, a boundary species with 'hasOnlySubstanceUnits=true' is in a non-unity compartment, and changes according to a rate rule.  This test ensures that simulators apply the rate rule to the species amount, and not to the species concentration.  This model is exactly the same as test 1219, but the Species initial value is set by its concentration, not its amount.

The model contains:
* 1 species (S1)
* 1 compartment (c)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0.6$ | variable |
| Initial volume of compartment 'c' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


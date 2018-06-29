(*

category:        Test
synopsis:        Testing the rateOf csymbol together with the hasOnlySubstanceUnits attribute
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, RateRule, Species
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, species S1 in non-unity compartment C is affected by a rate rule, which affects how its amount changes, because it is set hasOnlySubstanceUnits=true.  Similarly, the rateOf csymbol tracks how the amount of S1 changes in time instead of how its concentration changes in time.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $2$ |
| Assignment | x | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter x | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

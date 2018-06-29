(*

category:        Test
synopsis:        Testing the rateOf csymbol together with the hasOnlySubstanceUnits attribute
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, species S1 in non-unity compartment C is affected by a reaction, which affects how its amount changes.  Because it is set hasOnlySubstanceUnits=true, the rateOf csymbol tracks how the amount of S1 changes in time (i.e. the reaction rate), instead of how its concentration changes in time.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $2$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | x | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter x | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

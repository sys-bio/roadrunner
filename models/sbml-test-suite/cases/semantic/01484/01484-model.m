(*

category:        Test
synopsis:        An algebraic rule with a 'rateOf' csymbol.
componentTags:   AlgebraicRule, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, we have an algebraic rule that uses the 'rateOf' function.

The model contains:
* 1 species (S1)
* 2 parameters (conversionFactor, P1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $2$ |]

The conversion factor for the species 'S1' is 'conversionFactor'


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $P1 - S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter conversionFactor | $3$ | constant |
| Initial value of parameter P1 | $unknown$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

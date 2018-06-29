(*

category:        Test
synopsis:        A model with a rateOf, delay, and model conversion factor.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a reaction affects the level of two species, modified by a conversion factor.  This affects the 'rateOf' function, which in turn is used in a delay function.

The model contains:
* 2 species (S1, S2)
* 3 parameters (P0, P1, conversion)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]

The conversion factor for the model is 'conversion'


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |
| Assignment | P1 | $delay(rateOf(S1), 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter conversion | $2$ | constant |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial value of parameter P1 | $delay(rateOf(S1), 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

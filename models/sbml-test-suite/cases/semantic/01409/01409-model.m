(*

category:        Test
synopsis:        A model with a rateOf, delay, and model and species conversion factors.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a reaction affects the level of two species, one modified by a model conversion factor and the other by a species conversion factor.  This affects the 'rateOf' functions, which in turn are used in delay functions.

The model contains:
* 2 species (S1, S2)
* 6 parameters (P0, P1, P2, P3, sconversion, mconversion)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]

The conversion factor for the model is 'mconversion'
The conversion factor for the species 'S1' is 'sconversion'


There are 4 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |
| Assignment | P1 | $delay(rateOf(S1), 1)$ |
| Assignment | P2 | $rateOf(S2)$ |
| Assignment | P3 | $delay(rateOf(S2), 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter sconversion | $2$ | constant |
| Initial value of parameter mconversion | $3$ | constant |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial value of parameter P1 | $delay(rateOf(S1), 1)$ | variable |
| Initial value of parameter P2 | $rateOf(S2)$ | variable |
| Initial value of parameter P3 | $delay(rateOf(S2), 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

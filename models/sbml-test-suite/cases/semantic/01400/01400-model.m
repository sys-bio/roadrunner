(*

category:        Test
synopsis:        A delay of a rateOf.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

In this model, we test the calculation of the delay of the result of a 'rateOf' csymbol.

The model contains:
* 2 species (S1, S2)
* 2 parameters (P0, P1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |
| Assignment | P1 | $delay(rateOf(S1), 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial value of parameter P1 | $delay(rateOf(S1), 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

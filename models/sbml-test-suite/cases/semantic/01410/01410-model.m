(*

category:        Test
synopsis:        A combination of model and species conversion factors and the delay function.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, the species are affected by a reaction, and one of them is further affected by a model conversion factor, while the other by a species conversion factor.  The parameters P0 and P1 ensure that the delay functions incorporate the conversion factors into their own return value.

The model contains:
* 2 species (S1, S2)
* 4 parameters (P0, P1, sconversion, mconversion)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]

The conversion factor for the model is 'mconversion'
The conversion factor for the species 'S1' is 'sconversion'


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $delay(S1, 1)$ |
| Assignment | P1 | $delay(S2, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter sconversion | $2$ | constant |
| Initial value of parameter mconversion | $3$ | constant |
| Initial value of parameter P0 | $delay(S1, 1)$ | variable |
| Initial value of parameter P1 | $delay(S2, 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

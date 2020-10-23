(*

category:        Test
synopsis:        A combination of a species and model conversion factors and algebraic rules.
componentTags:   AlgebraicRule, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, the species are affected by a reaction, one of which is affected by a species conversion factor, and the other of which is affected by the model conversion factor.  The parameter P0 and P1 equal S1 and S2, respectively, through algebraic rules.

The model contains:
* 2 species (S1, S2)
* 4 parameters (P0, P1, mconversion, sconversion)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]

The conversion factor for the model is 'mconversion'
The conversion factor for the species 'S1' is 'sconversion'


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $P0 - S1$ |
| Algebraic | $0$ | $P1 - S2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter mconversion | $3$ | constant |
| Initial value of parameter sconversion | $2$ | constant |
| Initial value of parameter P0 | $unknown$ | variable |
| Initial value of parameter P1 | $unknown$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

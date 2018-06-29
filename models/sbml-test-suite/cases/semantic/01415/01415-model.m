(*

category:        Test
synopsis:        A delay function in a kinetic law that uses a local parameter as an argument.
componentTags:   CSymbolDelay, Compartment, Parameter, RateRule, Reaction, Species
testTags:        Amount, LocalParameters, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, kinetic law uses a delay function with a local parameter as an argument for the delay length.  It shadows a global parameter with the same ID but a different value.

The model contains:
* 1 species (S1)
* 2 parameters (P1, k)
* 1 local parameter (J0.k)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> | $0.1 * delay(P1, k)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial value of parameter P1 | $3$ | variable |
| Initial value of parameter k | $0$ | variable |
| Initial value of local parameter 'J0.k' | $1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

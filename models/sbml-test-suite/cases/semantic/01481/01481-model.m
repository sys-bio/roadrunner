(*

category:        Test
synopsis:        Stoichiometry math with a delay.
componentTags:   CSymbolDelay, Compartment, Reaction, Species, StoichiometryMath
testTags:        Amount, AssignedVariableStoichiometry, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Numeric
packagesPresent: 

This model tests stoichiometry math with a delay in it.

The model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (S1_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_sr S1 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_sr

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_sr' | $delay(S1, 1)$ | variable |]

*)

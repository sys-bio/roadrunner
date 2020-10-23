(*

category:        Test
synopsis:        Assigned stoichiometry with a delay.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

This model tests an assigned stoichiometry with a delay in it.

The model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (S1_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_sr S1 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_sr


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1_sr | $delay(S1, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_sr' | $delay(S1, 1)$ | variable |]

*)

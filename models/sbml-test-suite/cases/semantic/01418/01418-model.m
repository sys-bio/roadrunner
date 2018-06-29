(*

category:        Test
synopsis:        A delay function of a species reference.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a delay function references a (constant) species reference.

The model contains:
* 1 species (S1)
* 1 parameter (P1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> | $0.1 * S1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $delay(S1_stoich, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial value of parameter P1 | $delay(S1_stoich, 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

(*

category:        Test
synopsis:        A delay function of a species reference.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, InitialValueReassigned, NonConstantParameter, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a delay function references a (non-constant) species reference.

The model contains:
* 1 species (S1)
* 1 parameter (P1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1_stoich S1 -> | $0.1 * S1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $delay(S1_stoich, 1)$ |
| Rate | S1_stoich | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial value of parameter P1 | $delay(S1_stoich, 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

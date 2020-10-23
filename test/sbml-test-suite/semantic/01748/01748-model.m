(*

category:        Test
synopsis:        A variable stoichiometry for a substance-only species.
componentTags:   AssignmentRule, CSymbolTime, Compartment, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, HasOnlySubstanceUnits, InitialValueReassigned, NonUnityCompartment, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a variable stoichiometry is externally set for a species set hasOnlySubstanceUnits=true.

The model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1_stoich | $time / 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |
| Initial value of species reference 'S1_stoich' | $time / 10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

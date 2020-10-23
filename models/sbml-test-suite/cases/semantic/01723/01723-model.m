(*

category:        Test
synopsis:        A stoichiometry change rate set with reference to avogadro.
componentTags:   CSymbolAvogadro, Compartment, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the rate rule for a reaction stoichiometry is set to be equal to avogadro/10^24.

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
| Rate | S1_stoich | $avogadro / 1e24$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

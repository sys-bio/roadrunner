(*

category:        Test
synopsis:        A stoichiometry set with reference to avogadro.
componentTags:   CSymbolAvogadro, Compartment, InitialAssignment, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the stoichiometry of a reaction is set to be equal to avogadro/10^23.

The model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $avogadro / 1e23$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

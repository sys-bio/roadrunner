(*

category:        Test
synopsis:        A set stoichiometry for a substance-only species.
componentTags:   Compartment, Reaction, Species, StoichiometryMath
testTags:        Amount, AssignedConstantStoichiometry, HasOnlySubstanceUnits, NonUnityCompartment, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In this model, a stoichiometry is set via StoichiometryMath for a species set hasOnlySubstanceUnits=true.

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
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |
| Initial value of species reference 'S1_stoich' | $1.5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

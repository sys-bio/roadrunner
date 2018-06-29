(*

category:        Test
synopsis:        A species reference to a substance-only species.
componentTags:   Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, NonUnityCompartment, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an initial assignment uses a species reference for a substance-only species.

The model contains:
* 2 species (S1, S2)
* 1 parameter (k0)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: 2S1 -> S2 | $0.01$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter k0 | $S1_stoich * 2$ | constant |
| Initial volume of compartment 'C' | $2$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

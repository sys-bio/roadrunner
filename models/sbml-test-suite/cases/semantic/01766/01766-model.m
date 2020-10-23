(*

category:        Test
synopsis:        A local parameter shadowing a species reference.
componentTags:   Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, LocalParameters, NonUnityCompartment, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter shadows a species reference.  That ID is used in an initial assignment, and refers to the species reference.  A modified version of 1656 that is legal in l3v2 (which doesn't allow local parameters to have the ID of a species reference in the same reaction).

The model contains:
* 2 species (S1, S2)
* 1 parameter (k0)
* 1 local parameter (J1.S1_stoich)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: 2S1 -> | $0.01$ |
| J1: -> S2 | $S1_stoich$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter k0 | $S1_stoich * 2$ | constant |
| Initial value of local parameter 'J1.S1_stoich' | $0.01$ | constant |
| Initial volume of compartment 'C' | $2$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

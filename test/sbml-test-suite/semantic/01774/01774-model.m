(*

category:        Test
synopsis:        A local parameter shadowing a species reference in a kinetic law.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, HasOnlySubstanceUnits, LocalParameters, NonUnityCompartment, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter shadows a species reference.  That ID is used in a kinetic law, and refers to the species reference.  Similar to 1773, but the species reference and local parameter are both in different reactions from the use of the species reference in the kinetic law.

The model contains:
* 3 species (S1, S2, S3)
* 1 local parameter (J1.S1_stoich)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There are 3 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: 2S1 -> | $0.01$ |
| J1: -> S2 | $S1_stoich$ |
| J2: S3 -> | $S1_stoich / 20$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial amount of species S3 | $4$ | variable |
| Initial value of local parameter 'J1.S1_stoich' | $0.01$ | constant |
| Initial volume of compartment 'C' | $2$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

(*

category:        Test
synopsis:        A local parameter that shadows a species reference, next to another species reference.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, LocalParameters, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter shadows a species reference, in a formula that also includes a different species reference.

The model contains:
* 2 species (S1, S2)
* 1 local parameter (J0.S2_stoich)
* 1 compartment (C)
* 2 species references (S1_stoich, S2_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $S1_stoich * S2_stoich$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of local parameter 'J0.S2_stoich' | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $1$ | constant |
| Initial value of species reference 'S2_stoich' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

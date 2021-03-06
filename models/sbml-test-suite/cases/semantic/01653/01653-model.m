(*

category:        Test
synopsis:        A model conversion factor set with a species reference.
componentTags:   Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a model conversion factor is set with a species reference value.

The model contains:
* 2 species (S1, S2)
* 1 parameter (m_cf)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: 2S1 -> S2 | $0.01$ |]

The conversion factor for the model is 'm_cf'

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter m_cf | $S1_stoich * 3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

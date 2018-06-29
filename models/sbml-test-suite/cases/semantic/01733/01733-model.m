(*

category:        Test
synopsis:        Model and species conversion factors affecting a reaction with assigned stoichiometries and boundary conditions.
componentTags:   Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, BoundaryCondition, ConversionFactors, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, model and species conversion factors affect a reaction with assigned stoichiometries and boundary conditions, changing the levels of S1 and S2, but not S3 and S4.

The model contains:
* 4 species (S1, S2, S3, S4)
* 2 parameters (m_cf, s1_cf)
* 1 compartment (C)
* 4 species references (S3_stoich, S1_stoich, S2_stoich, S4_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S3_stoich S3 + S1_stoich S1 -> S2_stoich S2 + S4_stoich S4 | $0.01$ |]
Note:  the following stoichiometries are set separately:  S3_stoich, S1_stoich, S2_stoich, S4_stoich

The conversion factor for the model is 'm_cf'
The conversion factor for the species 'S1' is 's1_cf'
The conversion factor for the species 'S3' is 's1_cf'

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial amount of species S3 | $2$ | variable |
| Initial amount of species S4 | $3$ | variable |
| Initial value of parameter m_cf | $3$ | constant |
| Initial value of parameter s1_cf | $5$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S3_stoich' | $1.5$ | constant |
| Initial value of species reference 'S1_stoich' | $1.5$ | constant |
| Initial value of species reference 'S2_stoich' | $1.5$ | constant |
| Initial value of species reference 'S4_stoich' | $1.5$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

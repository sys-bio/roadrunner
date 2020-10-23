(*

category:        Test
synopsis:        A combination test of almost everything but events.
componentTags:   AssignmentRule, CSymbolAvogadro, CSymbolTime, Compartment, FunctionDefinition, InitialAssignment, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, AssignedVariableStoichiometry, BoundaryCondition, ConstantSpecies, ConversionFactors, HasOnlySubstanceUnits, InitialValueReassigned, LocalParameters, MultiCompartment, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A stripped-down version of model 1000, without events.  Designed to test a lot of thing in concert.

The model contains:
* 4 species (S1, S2, S3, S4)
* 8 parameters (kavo, k1, k3, k4, k5, k2, conversion1, conversion2)
* 2 compartments (comp, comp2)

It also contains 1 function definition(s):
; kinetics: $(-a + b + c + d + e + f + g + h + i + j) / 10$

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| _J0: S3 + S1ref S1 -> S2ref S2 | $kinetics(k1, k2, k3, k4, k5, S1, S1ref, S3, S4, S2)$ |]
Note:  the following stoichiometries are set separately:  S1ref, S2ref


There are 4 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | comp | $1$ |
| Assignment | comp2 | $k4$ |
| Rate | k4 | $time$ |
| Assignment | S2ref | $k1 * S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S4 | $2$ | constant |
| Initial amount of species S1 | $1$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial amount of species S3 | $4$ | variable |
| Initial value of parameter kavo | $avogadro / 6.022000e+023$ | constant |
| Initial value of parameter k1 | $1.1$ | constant |
| Initial value of parameter k2 | $8.12$ | constant |
| Initial value of parameter conversion1 | $10$ | constant |
| Initial value of parameter conversion2 | $100$ | constant |
| Initial value of parameter k3 | $2.5$ | variable |
| Initial value of parameter k4 | $1$ | variable |
| Initial value of parameter k5 | $2.8$ | variable |
| Initial volume of compartment 'comp' | $5$ | variable |
| Initial volume of compartment 'comp2' | $k4$ | variable |]

*)


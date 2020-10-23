(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule.
componentTags:   AssignmentRule, Compartment, InitialAssignment, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, NonUnityStoichiometry, SpeciesReferenceInMath, VolumeConcentrationRates
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A species in a non-unity compartment under control of a rate rule, and the species is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by the rate of change.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.  One of the rate rules gets its value from a species reference.

The model contains:
* 2 species (S1, S2)
* 1 parameter (x)
* 1 compartment (C1)
* 1 species reference (S2_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S2_sr S2 | $1$ |]
Note:  the following stoichiometries are set separately:  S2_sr


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $S2_sr$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |
| Initial value of species reference 'S2_sr' | $0.4$ | constant |]

*)

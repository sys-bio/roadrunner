(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule,with an initial assignment to the species.
componentTags:   AssignmentRule, Compartment, InitialAssignment, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, VolumeConcentrationRates
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A species in a non-unity compartment under control of a rate rule, and the species is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by the rate of change.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.  Its concentration is set with an initial assignment (not its amount).

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0.4$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3 / 2$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

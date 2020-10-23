(*

category:        Test
synopsis:        Testing the rateOf csymbol together with a non-unity compartment
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, RateRule, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a species in a non-unity compartment is affected by a reaction, but is set hasOnlySubstanceUnits=false, meaning that outside of reactions, it is to be treated as a concentration.  This includes the 'rateOf' csymbol, meaning that 'rateOf(S1)' means the rate of change of the concentration of S1, not the rate of change of the amount of S1.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $2$ |
| Assignment | x | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter x | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

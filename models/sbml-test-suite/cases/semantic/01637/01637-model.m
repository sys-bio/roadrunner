(*

category:        Test
synopsis:        A rate rule for a species reference that has the same ID as a local parameter.
componentTags:   Compartment, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, LocalParameters, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, a local parameter has the same ID as a species reference, which itself has a rate rule for it.

The model contains:
* 1 species (S1)
* 1 local parameter (J0.k0)
* 1 compartment (C)
* 1 species reference (k0)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> k0 S1 | $1 / k0$ |]
Note:  the following stoichiometries are set separately:  k0


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k0 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of local parameter 'J0.k0' | $3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'k0' | $1$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

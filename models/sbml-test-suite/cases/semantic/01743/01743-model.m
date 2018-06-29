(*

category:        Test
synopsis:        A stoichiometry that changes due to a function definition.
componentTags:   Compartment, FunctionDefinition, Parameter, RateRule, Reaction, Species, StoichiometryMath
testTags:        Amount, AssignedVariableStoichiometry, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In this model, a stoichiometry is changed due to a rate rule that calls a function definition. This model is the same as 1742, but uses the StoichiometryMath of level 2 SBML.

The model contains:
* 1 species (S1)
* 1 parameter (parameterId_0)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | parameterId_0 | $getOnePointFive()$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | getOnePointFive |  | $1.5$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of parameter parameterId_0 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $parameterId_0$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

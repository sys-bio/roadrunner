(*

category:        Test
synopsis:        A variable stoichiometry with a shadowing local parameter.
componentTags:   AssignmentRule, CSymbolTime, Compartment, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, InitialValueReassigned, LocalParameters, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, a variable stoichiometry has a local parameter that shadows its ID.

The model contains:
* 1 species (S1)
* 1 local parameter (J0.S1_stoich)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $S1_stoich$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1_stoich | $time / 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial value of local parameter 'J0.S1_stoich' | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $time / 10$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

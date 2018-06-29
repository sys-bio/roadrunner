(*

category:        Test
synopsis:        A species reference with an empty assignment rule.
componentTags:   AssignmentRule, Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, InitialValueReassigned, NoMathML, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an initial assignment uses a species reference which has an assignment rule, but that assignment rule has no MathML, leaving it with its original value.

The model contains:
* 2 species (S1, S2)
* 1 parameter (k0)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1_stoich S1 -> S2 | $0.01$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1_stoich | $$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter k0 | $S1_stoich * 2$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

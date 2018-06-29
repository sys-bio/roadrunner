(*

category:        Test
synopsis:        A fast reaction with a species reference used in an assignment rule.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:        Amount, FastReaction, InitialValueReassigned, NonUnityStoichiometry, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, a fast reaction contains a species reference which is itself referenced in a parameter's assignment rule.

The model contains:
* 2 species (S1, S2)
* 1 parameter (P1)
* 1 compartment (C)
* 1 species reference (S2_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> 2S2 | $S1 * 0.1$ | fast |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $S2_stoich$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter P1 | $S2_stoich$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S2_stoich' | $2$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

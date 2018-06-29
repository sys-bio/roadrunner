(*

category:        Test
synopsis:        A delay of species from a fast reaction.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Parameter, Reaction, Species
testTags:        Amount, FastReaction, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, the delayed value of S1 and S2 are assigned to P1 and P2, with S1 and S2 participating in a fast reaction, which goes to completion at t0, but not before t0.

The model contains:
* 2 species (S1, S2)
* 2 parameters (P1, P2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> S2 | $S1 * 1$ | fast |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $delay(S1, 1.5)$ |
| Assignment | P2 | $delay(S2, 1.5)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter P1 | $delay(S1, 1.5)$ | variable |
| Initial value of parameter P2 | $delay(S2, 1.5)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

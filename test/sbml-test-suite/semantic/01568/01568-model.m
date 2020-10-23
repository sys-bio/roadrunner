(*

category:        Test
synopsis:        A fast reaction whose rate depends on an assignment rule.
componentTags:   AssignmentRule, CSymbolTime, Compartment, Parameter, Reaction, Species
testTags:        Amount, FastReaction, InitialValueReassigned, NonConstantParameter, ReversibleReaction
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

The model tests a fast reaction which is positive or negative depending on the results of an assignment rule.

The model contains:
* 2 species (S1, S2)
* 1 parameter (k)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> S2 | $piecewise(S1 * k, k > 0, S2 * k)$ | fast |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | k | $5.5 - time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial value of parameter k | $5.5 - time$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

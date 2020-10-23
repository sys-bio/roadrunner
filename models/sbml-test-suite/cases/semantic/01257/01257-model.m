(*

category:        Test
synopsis:        Two rateOf csymbols for a species and parameter.
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, RateRule, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

Two rateOf csymbols are used here in an assignment rule: one that references a species changing due to a reactions with a non-constant kinetic law, and one that references a parameter changing due to a rate rule.

The model contains:
* 1 species (S1)
* 2 parameters (p1, p2)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $0.01 * S1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |
| Assignment | p2 | $rateOf(S1) + rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $rateOf(S1) + rateOf(p1)$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

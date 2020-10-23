(*

category:        Test
synopsis:        The rateOf csymbol for a species that changes due to a reaction with a non-constant kinetic law.
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 The rateOf csymbol is used here in an assignment rule that references a species changing due to a reactions with a non-constant kinetic law.

The model contains:
* 1 species (S1)
* 1 parameter (p2)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $0.01 * S1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p2 | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter p2 | $rateOf(S1)$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

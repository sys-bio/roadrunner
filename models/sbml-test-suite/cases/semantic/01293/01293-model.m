(*

category:        Test
synopsis:        A rateOf csymbol in an event trigger.
componentTags:   CSymbolRateOf, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

A rateOf csymbol is used here in an event trigger that assigns to p2.  This model is the same as model 01261, with S1 changing as a species in a reaction, instead of in a rate rule.

The model contains:
* 1 species (S1)
* 1 parameter (p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $0.01 * S1$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $(100 * rateOf(S1)) > 1.05$ | $p2 = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

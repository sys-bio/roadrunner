(*

category:        Test
synopsis:        A variable rateOf csymbol in several contexts at once.
componentTags:   CSymbolRateOf, CSymbolTime, Compartment, EventPriority, EventWithDelay, Parameter, RateRule, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

A variable rateOf csymbol is used in all valid MathML contexts.

The model contains:
* 1 species (S1)
* 2 parameters (p1, p2)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $rateOf(p1)$ |]


There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Delay*  | *Event Assignments* |
| _E0 | $time > 4.5$ | $0.02$ | $rateOf(p1)$ | $p2 = rateOf(p1) + 2$ |
| _E1 | $time > 4.5$ | $rateOf(p1)$ | $rateOf(p1)$ | $p2 = rateOf(p1) + 5$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $10$ | variable |
| Initial value of parameter p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

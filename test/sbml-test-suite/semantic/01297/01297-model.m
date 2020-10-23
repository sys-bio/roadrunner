(*

category:        Test
synopsis:        A variable rateOf csymbol in an event assignment.
componentTags:   CSymbolRateOf, CSymbolTime, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

A variable rateOf csymbols is used here in an event assignment for p2.  This is exactly the same model as 01266, with p1 changing as a species in a reaction, instead of in a rate rule.

The model contains:
* 1 species (p1)
* 1 parameter (p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> p1 | $0.01 * p1$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $p2 = rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

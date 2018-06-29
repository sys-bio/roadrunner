(*

category:        Test
synopsis:        A rateOf csymbol in an event priority.
componentTags:   CSymbolRateOf, CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A rateOf csymbol is used here in an event priority that assigns to p2.  This is exactly the same model as 01262, with p1 changing as a species in a reaction, instead of in a rate rule.

The model contains:
* 1 species (p1)
* 1 parameter (p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> p1 | $1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $1.01$ | $p2 = 5$ |
| E1 | $time > 5.5$ | $rateOf(p1)$ | $p2 = 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species p1 | $1$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

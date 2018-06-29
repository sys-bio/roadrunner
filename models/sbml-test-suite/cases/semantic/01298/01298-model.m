(*

category:        Test
synopsis:        A variable rateOf csymbol in an event priority.
componentTags:   CSymbolRateOf, CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 A variable rateOf csymbol is used here in an event priority that assigns to p2.  At the time the event triggers, the priority is greater than .021, meaning that it fires first, and E0 fires second, leaving the value of p2 at 5.  This is exactly the same model as 01267, with p1 changing as a species in a reaction, instead of in a rate rule.

The model contains:
* 1 species (p1)
* 1 parameter (p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> p1 | $0.01 * p1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $0.021$ | $p2 = 5$ |
| E1 | $time > 5.5$ | $rateOf(p1)$ | $p2 = 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

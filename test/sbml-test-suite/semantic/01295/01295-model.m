(*

category:        Test
synopsis:        A rateOf csymbol in an event delay.
componentTags:   CSymbolRateOf, CSymbolTime, Compartment, EventWithDelay, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A rateOf csymbol is used here in an event delay that assigns to p2. This is exactly the same model as 01263, with p1 changing as a species in a reaction, instead of in a rate rule. 

The model contains:
* 1 species (p1)
* 1 parameter (p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> p1 | $1$ |]


There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time > 4.5$ | $rateOf(p1)$ | $p2 = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species p1 | $1$ | variable |
| Initial value of parameter p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

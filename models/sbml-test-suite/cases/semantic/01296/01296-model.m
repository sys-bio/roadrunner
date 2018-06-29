(*

category:        Test
synopsis:        A variable rateOf csymbol in a kinetic law.
componentTags:   CSymbolRateOf, Compartment, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A variable rateOf csymbols is used here in a kinetic law, defining the rate of change of p2 the same as the rate of change of p1.  This is exactly the same model as 01264, with p1 and p2 changing as species in reactions, instead of in a rate rule.

The model contains:
* 2 species (p1, p2)
* 1 compartment (C)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> p1 | $0.01 * p1$ |
| J1: -> p2 | $rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species p1 | $2$ | variable |
| Initial concentration of species p2 | $10$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)

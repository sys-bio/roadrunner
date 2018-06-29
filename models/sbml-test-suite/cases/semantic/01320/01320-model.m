(*

category:        Test
synopsis:        Use of a csymbol whose putative ID shadows another parameter.
componentTags:   AssignmentRule, CSymbolDelay, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model the 'delay' csymbol is given the ID of a different parameter.  This ID should be ignored.

The model contains:
* 1 species (S1)
* 2 parameters (p1, p2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p2 | $delay(S1, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter p1 | $5$ | variable |
| Initial value of parameter p2 | $delay(S1, 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

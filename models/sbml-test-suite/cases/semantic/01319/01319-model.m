(*

category:        Test
synopsis:        Use of a csymbol whose putative ID shadows another parameter.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolTime, Parameter
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model the 'delay' csymbol is given the ID of a different function.  This ID should be ignored.

The model contains:
* 3 parameters (p2, p3, p1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p2 | $1 + time$ |
| Assignment | p3 | $delay(p2, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | constant |
| Initial value of parameter p2 | $1 + time$ | variable |
| Initial value of parameter p3 | $delay(p2, 1)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

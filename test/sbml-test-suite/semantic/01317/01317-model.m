(*

category:        Test
synopsis:        Use of a csymbol whose putative ID shadows another parameter.
componentTags:   AssignmentRule, CSymbolTime, Parameter
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model the 'time' csymbol is given the ID of another parameter.  This ID should be ignored.

The model contains:
* 2 parameters (p1, p2)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p2 | $1 + time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | constant |
| Initial value of parameter p2 | $1 + time$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

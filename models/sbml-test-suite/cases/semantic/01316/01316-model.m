(*

category:        Test
synopsis:        Use of a csymbol whose putative ID shadows another parameter.
componentTags:   CSymbolAvogadro, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model the 'avogadro' csymbol is given the ID of another parameter.  This ID should be ignored.

The model contains:
* 2 parameters (p1, p2)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | constant |
| Initial value of parameter p2 | $1 + avogadro / 1.000000e+24$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

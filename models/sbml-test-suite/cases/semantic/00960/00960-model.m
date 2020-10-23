(*

category:      Test
synopsis:      Four different ways of writing the avogadro number in an initial assignment.
componentTags: CSymbolAvogadro, InitialAssignment, Parameter
testTags:      InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model assigns avogadro's number as a csymbol to four parameters.  It tries to fool you by calling two of them 'time' and 'delay', and by not giving one a name at all.

To test that the value of avogadro is the same as is in the specification without requiring that you write out all 9 digits of precision, a fifth parameter is added with the value (avogadro - 6.0221e+23).

The model contains:
* 5 parameters (P1, P2, P3, P4, P5)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $avogadro$ | constant |
| Initial value of parameter P2 | $time$ | constant |
| Initial value of parameter P3 | $delay$ | constant |
| Initial value of parameter P4 | $$ | constant |
| Initial value of parameter P5 | $avogadro - 6.022100e+023$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


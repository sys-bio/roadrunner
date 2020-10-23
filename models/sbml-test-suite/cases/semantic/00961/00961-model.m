(*

category:      Test
synopsis:      Four different ways of writing the avogadro number in an assignment rule.
componentTags: AssignmentRule, CSymbolAvogadro, Parameter
testTags:      InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model assigns avogadro's number as a csymbol to four parameters in an assignment rule.  It tries to fool you by calling two of them 'time' and 'delay', and by not giving one a name at all.

To test that the value of avogadro is the same as is in the specification without requiring that you write out all 9 digits of precision, a fifth parameter is added with the value (avogadro - 6.0221e+23).

The model contains:
* 5 parameters (P1, P2, P3, P4, P5)

There are 5 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $avogadro$ |
| Assignment | P2 | $time$ |
| Assignment | P3 | $delay$ |
| Assignment | P4 | $$ |
| Assignment | P5 | $avogadro - 6.022100e+023$ |]

 Note:  All parameters actually assigned the CSymbol 'avogadro', but the text is different for the first four cases.
 
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $avogadro$ | variable |
| Initial value of parameter P2 | $time$ | variable |
| Initial value of parameter P3 | $delay$ | variable |
| Initial value of parameter P4 | $$ | variable |
| Initial value of parameter P5 | $avogadro - 6.022100e+023$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


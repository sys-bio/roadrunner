(*

category:        Test
synopsis:        The rateOf csymbol with a constant parameter.
componentTags:   CSymbolRateOf, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

The rateOf csymbol is used here in an initial assignment that references a constant parameter.  Hence, the rate is zero.  The parameter in question is flagged 'constant=false', however, even though it does not actually change.

The model contains:
* 2 parameters (p1, p2)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $rateOf(p1)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

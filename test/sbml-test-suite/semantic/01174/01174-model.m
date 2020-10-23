(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor and an algebraic rule.
componentTags:   AlgebraicRule, CSymbolDelay, CSymbolTime, InitialAssignment, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model contains a submodel with an algebraic rule containing the csymbols 'time' and 'delay' , contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 3 parameters (timeconv, t3, t4)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $t4 - delay(t3, timeconv * (time / timeconv / 2))$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t3 | $time / timeconv + 3$ | constant |
| Initial value of parameter t4 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


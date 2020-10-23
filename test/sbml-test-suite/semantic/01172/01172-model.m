(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor and a rate rule.
componentTags:   CSymbolTime, Parameter, RateRule, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model contains a submodel with a rate rule in it, contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (timeconv, t1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | t1 | $(time / timeconv / t1 + 3) / timeconv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


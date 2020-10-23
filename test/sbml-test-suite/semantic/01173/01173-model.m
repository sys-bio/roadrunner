(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor and delay.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolTime, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model contains a submodel with the csymbol delay, contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 3 parameters (timeconv, t1, t3)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | t1 | $time / timeconv$ |
| Assignment | t3 | $delay(t1, timeconv * 3)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t1 | $time / timeconv$ | variable |
| Initial value of parameter t3 | $delay(t1, timeconv * 3)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


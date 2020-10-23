(*

category:        Test
synopsis:        A hierarchical model with a time conversion and a replacement conversion factor both affecting a rate rule.
componentTags:   CSymbolTime, Parameter, RateRule, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        NonConstantParameter, comp:ConversionFactor, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The rate rule in the submodel affects a parameter that has been replaced with a conversion factor, necessitating a conversion of the rate rule.  In addition, the submodel as a whole has a time conversion factor, so the rate rule must additionally be affected by that.  Just to go whole hog, the parameter itself is used in its own rate rule, as is the csymbol 'time'.

The 'flattened' version of this hierarchical model contains:
* 3 parameters (timeconv, paramconv, t1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | t1 | $(time / timeconv / (t1 / paramconv) + 3) * paramconv / timeconv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter paramconv | $0.01$ | constant |
| Initial value of parameter t1 | $1$ | variable |]

*)


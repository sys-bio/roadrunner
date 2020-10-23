(*

category:        Test
synopsis:        Simple non-required comp model.
componentTags:   Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This hierarchical model replaces the single parameter in the submodel, and therefore behaves as if the submodel did not exist.  The comp package 'required' attribute is therefore 'false', meaning that the mathematical meaning of the model is not changed by the presence of 'comp' elements.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (param1)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter param1 | $10.42$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


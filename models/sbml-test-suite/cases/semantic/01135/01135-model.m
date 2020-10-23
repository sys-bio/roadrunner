(*

category:        Test
synopsis:        Hierarchical model with chained replacedBys.
componentTags:   Parameter, comp:ModelDefinition, comp:ReplacedBy, comp:SBaseRef, comp:Submodel
testTags:        
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The parameter in the parent model is replaced by a parameter in the submodel, which in turn is replaced by a parameter in *its* submodel.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (p2)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p2 | $8$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


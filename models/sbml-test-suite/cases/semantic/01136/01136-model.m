(*

category:        Test
synopsis:        Chained replaced/replacedBy elements.
componentTags:   Parameter, comp:ModelDefinition, comp:ReplacedBy, comp:SBaseRef, comp:ReplacedElement, comp:Submodel
testTags:        
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 Despite starting off with five parameters at three levels, all aggregate parameters collapse down to a single parameter, as one set is replaced by parameters in models above them, and the other is replaced by parameters below them.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (p2)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p2 | $8$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


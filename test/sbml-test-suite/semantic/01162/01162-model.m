(*

category:        Test
synopsis:        A hierarchical model with a rate rule deleted via a port.
componentTags:   Parameter, comp:Deletion, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The rate rule for the parameter in the submodel is defined with a port, and the model that imports it sets the port as the target of a 'delete' construct.  This should, by proxy, delete the rate rule.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (p8)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p8 | $8$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


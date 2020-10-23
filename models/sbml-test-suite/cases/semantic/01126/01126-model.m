(*

category:        Test
synopsis:        A simple hierarchical model with a submodel parameter
componentTags:   Parameter, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

There are two parameters in this model, one in the parent model which replaces a paramter in the submodel, and a second parameter from the submodel that remains unreplaced.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (param1, submod1__subparam2)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter param1 | $10.42$ | constant |
| Initial value of parameter submod1__subparam2 | $6$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


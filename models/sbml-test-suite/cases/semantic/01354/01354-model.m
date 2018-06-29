(*

category:        Test
synopsis:        A replaced parameter whose ID is used in an initial assignment.
componentTags:   InitialAssignment, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced parameter ID is used in an initial assignment.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (J1, A__x)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter J1 | $5$ | constant |
| Initial value of parameter A__x | $J1 + 2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

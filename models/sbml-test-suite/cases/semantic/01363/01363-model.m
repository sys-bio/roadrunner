(*

category:        Test
synopsis:        A replaced compartment whose ID is used in an initial assignment.
componentTags:   Compartment, InitialAssignment, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonUnityCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced compartment ID is used in an initial assignment.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (A__x)
* 1 compartment (J1)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter A__x | $J1 + 2$ | constant |
| Initial volume of compartment 'J1' | $5$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

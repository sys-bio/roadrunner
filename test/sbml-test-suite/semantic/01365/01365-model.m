(*

category:        Test
synopsis:        A replaced compartment whose ID is used in an event priority.
componentTags:   CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        NonConstantParameter, NonUnityCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced compartment ID is used in an event priority.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (A__x)
* 1 compartment (J1)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| A__E0 | $time > 5.5$ | $J1$ | $A__x = 4$ |
| A__E1 | $time > 5.5$ | $4$ | $A__x = 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter A__x | $1$ | variable |
| Initial volume of compartment 'J1' | $5$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

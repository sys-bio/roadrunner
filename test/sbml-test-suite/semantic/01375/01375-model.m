(*

category:        Test
synopsis:        A replaced species whose ID is used in an event assignment.
componentTags:   CSymbolTime, Compartment, EventNoDelay, Parameter, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, BoundaryCondition, NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced species ID is used in an event assignment.

The 'flattened' version of this hierarchical model contains:
* 1 species (J1)
* 1 parameter (A__x)
* 1 compartment (A_C)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| A__E0 | $time > 5.5$ | $A__x = J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species J1 | $5$ | variable |
| Initial value of parameter A__x | $1$ | variable |
| Initial volume of compartment 'A_C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

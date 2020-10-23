(*

category:        Test
synopsis:        A replaced reaction whose ID is used in an event assignment.
componentTags:   CSymbolTime, Compartment, EventNoDelay, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced reaction ID is used in an event assignment.

The 'flattened' version of this hierarchical model contains:
* 1 species (S1)
* 1 parameter (A__x)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: -> S1 | $5$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| A__E0 | $time > 5.5$ | $A__x = J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial value of parameter A__x | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

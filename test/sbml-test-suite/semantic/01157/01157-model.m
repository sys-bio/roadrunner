(*

category:        Test
synopsis:        A hierarchical model that deletes an event assignment.
componentTags:   CSymbolTime, EventNoDelay, Parameter, comp:Deletion, comp:ModelDefinition, comp:Submodel
testTags:        NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The original event in the submodel contains two event assignments, one of which is deleted when the event's submodel is imported into the parent model.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (sub1__t1, sub1__t2)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Delay*  | *Event Assignments* |
| sub1__E0 | $gt(time, 3)$ | $10$ | $1 / time$ | $sub1__t2 = 5.5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter sub1__t1 | $1$ | variable |
| Initial value of parameter sub1__t2 | $1$ | variable |]

*)


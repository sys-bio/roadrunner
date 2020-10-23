(*

category:        Test
synopsis:        This hierarchical model deletes the delay of an event.
componentTags:   CSymbolTime, EventNoDelay, Parameter, comp:Deletion, comp:ModelDefinition, comp:Submodel
testTags:        NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The event in the submodel contains a delay, but this delay is deleted when the submodel is imported into the parent model, causing the event to fire instantly.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (sub1__t1, sub1__t2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| sub1__E0 | $gt(time, 3)$ | $sub1__t1 = 3.3$ |
|  |  |  | $sub1__t2 = 5.5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter sub1__t1 | $1$ | variable |
| Initial value of parameter sub1__t2 | $1$ | variable |]

*)


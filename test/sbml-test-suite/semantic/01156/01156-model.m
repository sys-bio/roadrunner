(*

category:        Test
synopsis:        A hierarchical model with a deleted event priority.
componentTags:   CSymbolTime, EventWithDelay, Parameter, comp:Deletion, comp:ModelDefinition, comp:Submodel
testTags:        NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 This model contains an event with a priority which is deleted when imported as part of a submodel.  It is actually impossible to test the behavior of a model that deletes a priority, because the behavior of an event without a priority has fewer restrictions on its behavior in a simulator than that same event with a priority, and allowing the event to behave as if it had any arbitrary priority (including the one deleted) is actually perfectly legal.  As a result, this test merely checks to ensure that the simulation occurred at all, despite containing that deletion construct.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (sub1__t1, sub1__t2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| sub1__E0 | $gt(time, 3)$ | $1 / time$ | $sub1__t1 = 3.3$ |
|  |  |  | $sub1__t2 = 5.5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter sub1__t1 | $1$ | variable |
| Initial value of parameter sub1__t2 | $1$ | variable |]

*)


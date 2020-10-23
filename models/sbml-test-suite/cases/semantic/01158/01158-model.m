(*

category:        Test
 synopsis:        A hierarchical model that deletes a priority, delay, and event assignment
componentTags:   CSymbolTime, EventNoDelay, Parameter, comp:Deletion, comp:ModelDefinition, comp:Submodel
testTags:        NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 In this model, a submodel event is greatly changed through deletions:  its delay, priority, and one of its event assignements are all deleted.  As in model 01156, the fact that the priority has been deleted cannot be tested numerically, but the other deletions all directly affect the expected output of the model.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (sub1__t1, sub1__t2)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| sub1__E0 | $gt(time, 3)$ | $sub1__t1 = 3.3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter sub1__t1 | $1$ | variable |
| Initial value of parameter sub1__t2 | $1$ | variable |]

*)


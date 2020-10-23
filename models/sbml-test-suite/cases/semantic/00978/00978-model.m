(*

category:      Test
synopsis:      Several events conspire within the same time step to trigger three events multiple times, with different outcomes.
componentTags: CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, EventIsPersistent, EventUsesAssignmentTimeValues, EventUsesTriggerTimeValues, NonConstantParameter
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains two events that trigger when the parameter multitrig increases to 
above 1.  The first is set 'persistent=true' and 'useValuesFromTriggerTime=false', and sets x=x+1, meaning that every time
it triggers, it ends up increasing x by one.  

The second is set 'persistent=true' but 'useValuesFromTriggerTime=true', meaning that it executes
multiple times, but each time uses the value of y from the time it first triggered, meaning
that y gets reset to one four times.

The third is set 'persistent=false' and 'useValuesFromTriggerTime=false',
and sets z=z+3, meaning that every time it triggers, it has the potential to increase z
by three, but if multitrig drops below 1 again, it will not actually do so.  In the end,
it only executes a single time.

The model also contains nine events with the same trigger and with decreasing priorities 
that alternate setting the value of the parameter 'multitrig' from zero to 2, then to 
zero, then back again, etc.  The priorities of all these events are higher than the first 
two, meaning that at a single time point, the first two events are triggered five times,
and 'de-triggered' four times.  Since the first event was persistent, x should increase
from zero to five, and since the second event was not persistent, y should increase from
zero to one.

For a different view of this model, see 00978-antimony.txt

The model contains:
* 4 parameters (multitrig, x, y, z)

There are 12 events:

[{width:45em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  |  *Use values from:*  | *Event Assignments* |
| _E0 | $time &geq; 0.99$ | $10$ | true | Trigger time | $multitrig = 2$ |
| _E1 | $time &geq; 0.99$ | $9$ | true | Trigger time | $multitrig = 0$ |
| _E2 | $time &geq; 0.99$ | $8$ | true | Trigger time | $multitrig = 2$ |
| _E3 | $time &geq; 0.99$ | $7$ | true | Trigger time | $multitrig = 0$ |
| _E4 | $time &geq; 0.99$ | $6$ | true | Trigger time | $multitrig = 2$ |
| _E5 | $time &geq; 0.99$ | $5$ | true | Trigger time | $multitrig = 0$ |
| _E6 | $time &geq; 0.99$ | $4$ | true | Trigger time | $multitrig = 2$ |
| _E7 | $time &geq; 0.99$ | $3$ | true | Trigger time | $multitrig = 0$ |
| _E8 | $time &geq; 0.99$ | $2$ | true | Trigger time | $multitrig = 2$ |
| _E9 | $multitrig > 1$ | $1$ | true | Assignment time | $x = x + 1$ |
| _E10 | $multitrig > 1$ | $1$ | true | Trigger time | $y = y + 1$ |
| _E11 | $multitrig > 1$ | $1$ | false | Assignment time | $z = z + 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter multitrig | $0$ | variable |
| Initial value of parameter x | $0$ | variable |
| Initial value of parameter y | $0$ | variable |
| Initial value of parameter z | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


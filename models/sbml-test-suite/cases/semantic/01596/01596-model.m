(*

category:        Test
synopsis:        Uncommon MathML in an event that fires at t0.
componentTags:   AssignmentRule, CSymbolTime, EventNoDelay, Parameter
testTags:        EventT0Firing, InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model tests the use of an uncommon mathematical construct in an event that fires at t0.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $cosh(time) > 0.5$ | false | $P2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $cosh(time)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $cosh(time)$ | variable |
| Initial value of parameter P2 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

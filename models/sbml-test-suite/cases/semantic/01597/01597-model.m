(*

category:        Test
synopsis:        Uncommon MathML in a delayed event with trigger time values.
componentTags:   AssignmentRule, CSymbolTime, EventWithDelay, Parameter
testTags:        EventUsesTriggerTimeValues, InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model tests the use of an uncommon mathematical construct in a delayed event with trigger time values.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time > 4.5$ | $1$ | $P2 = cosh(time)$ |]


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

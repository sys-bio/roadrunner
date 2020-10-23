(*

category:      Test
synopsis:      A fast and slow reaction, working against each other, with the fast reaction turned on and off by Events.
componentTags: Compartment, EventWithDelay, EventNoDelay, Parameter, Reaction, Species
testTags:      Amount, EventIsPersistent, FastReaction, NonConstantParameter, HasOnlySubstanceUnits
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:   Analytic

 After B is converted to A by the slow reaction for a while, eventually the two events trigger, setting the reaction rate of the fast reaction to 1*A--consuming all of A and turning it back into B.  Then it turns back off, and B is once again slowly converted to A.  This would cycle again, but we only test one cycle.


The model contains:
* 2 species (A, B)
* 2 parameters (k1, k2)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| B -> A | $k1$ | slow |
| A -> B | $k2 * A$ | fast |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $A &geq; 0.99$ | $0$ | $k2 = 1$ |
| E1 | $A &geq; 0.99$ | $0.1$ | $k2 = 0$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species A | $0$ | variable |
| Initial amount of species B | $2$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial value of parameter k2 | $0$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

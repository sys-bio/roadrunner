(*

category:      Test
synopsis:      Single forward reaction with two species in one compartment and three events
componentTags: CSymbolTime, Compartment, EventNoDelay, EventPriority, Species
testTags:      Amount, EventIsNotPersistent, EventIsPersistent
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model contains three events that assign value to species S1 and S2 defined below.

The events will execute simultaneously and must be executed in order determined by
the Priority. However, since Event2 is not persistent, the execution of Event1
causes the Trigger of Event2 to return to 'false'. It must therefore be removed
from the list of pending events.


The model contains:
* 2 species (S1, S2)
* 1 compartment (C)

There are 3 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| A | $and(geq(time, 0.99), lt(S1, 0.5))$ | $10$ | true | $S1 = 1$ |
|  |  |  |  | $S2 = 0$ |
| B | $and(geq(time, 0.99), lt(S1, 0.5))$ | $8$ | false | $S1 = 2$ |
|  |  |  |  | $S2 = 1$ |
| C1 | $and(geq(time, 0.99), lt(S1, 0.5))$ | $9$ | true | $S1 = 3$ |
|  |  |  |  | $S2 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $0$ | variable |
| Initial amount of species S2 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


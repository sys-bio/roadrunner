(*

category:      Test
synopsis:      Single forward reaction with two species in one compartment and an event
componentTags: CSymbolTime, Compartment, EventNoDelay, EventPriority, Species
testTags:      Amount, EventIsPersistent
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The Events will execute simultaneously and must be executed in order determined by
the Priority. Since the priority of all remaining events must be re-calculated
after each event is executed, the order in which the events execute changes after
Event1.


The model contains:
* 2 species (S1, S2)
* 1 compartment (C)

There are 3 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| A | $time &geq; 0.99$ | $10$ | $S1 = 1$ |
|  |  |  | $S2 = 0$ |
| B | $time &geq; 0.99$ | $2 * S2$ | $S1 = 2$ |
|  |  |  | $S2 = 1$ |
| C1 | $time &geq; 0.99$ | $2 * S1$ | $S1 = 3$ |
|  |  |  | $S2 = 2$ |]

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


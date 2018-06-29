(*

category:        Test
synopsis:        Competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure the two events are not exactly evenly distributed.  NOTE:  STOCHASTIC TEST. Your software may fail periodically; it is only supposed to succeed in the majority of cases.
componentTags:   AssignmentRule, CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, RateRule, Species
testTags:        EventIsNotPersistent, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, RandomEventExecution, VolumeConcentrationRates
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 This model contains two events with the same trigger, the same effective priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the stoichiometries Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.  A species S2 is controlled by a rate rule for its concentration based on Q-R, which should be close to zero for the duration of the simulation.  Its compartment is also under control of a rate rule, making the amount of S2 dependent on those two different rate rules.  Three other parameters (errorHigh, errorLow, and errorS2), check to make sure neither Q nor R are chosen more or less frequently than the other, and that the species S2 doesn't get too extreme (the reaction in which it appears should be balanced, on average).  Note:  The error parameters are a stochastic test, and may not always remain at '0' for all runs.  If your software fails, try running it again with a new random number seed, and it may succeed.  The values were chosen to be reasonable in the vast majority of cases, but still high enough to reveal problems in software that tends to pick both events either evenly or in an unbalanced manner.

The model contains:
* 1 species (S2)
* 9 parameters (S, Q, R, reset, maxdiff, errorLow, errorHigh, errorS2, S2_det)
* 1 compartment (C1)

There are 6 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $reset >= 0.01$ | $1$ | false | $reset = 0$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $reset >= 0.01$ | $1$ | false | $reset = 0$ |
|  |  |  |  | $R = R + 0.01$ |
| maxcheck | $abs(Q - R) > maxdiff$ | (unset) | true | $maxdiff = abs(Q - R)$ |
| error_check | $(time >= 99) && (maxdiff < 0.2)$ | (unset) | true | $errorLow = 1$ |
| error_check2 | $maxdiff >= 4$ | (unset) | true | $errorHigh = 1$ |
| error_check3 | $abs(S2) >= 250$ | (unset) | true | $errorS2 = 1$ |]


There are 4 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S2 | $Q - R$ |
| Rate | C1 | $0.002$ |
| Assignment | S | $Q + R$ |
| Rate | reset | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S2 | $0$ | variable |
| Initial value of parameter S | $Q + R$ | variable |
| Initial value of parameter Q | $1$ | variable |
| Initial value of parameter R | $1$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter maxdiff | $0$ | variable |
| Initial value of parameter errorLow | $0$ | variable |
| Initial value of parameter errorHigh | $0$ | variable |
| Initial value of parameter errorS2 | $0$ | variable |
| Initial value of parameter S2_det | $0$ | variable |
| Initial volume of compartment 'C1' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

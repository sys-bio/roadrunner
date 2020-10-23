(* 

category:      Test
synopsis:      Single forward reaction with two species in one compartment and an event
componentTags: Compartment, CSymbolTime, Species, Reaction, Parameter, EventNoDelay
testTags:      Amount, EventT0Firing
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "C".  There are two
species called S1 and S2 and one parameter called k1.  The model contains
one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$ |]

The model contains one event that assigns value to species S1 defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $t >= 0$ | $-$   | $S1 = 1$    |]
 
The initialValue of the Trigger is set to 'true'.  Thus at t = 0 the Trigger
does not transition from 'false' to 'true' and thus the event is not fired.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$1.5 \x 10^-4$  |mole           |
|Initial amount of S2                |$0$              |mole           |
|Value of parameter k1               |$1$              |second^-1^     |
|Volume of compartment "C" |$1$              |litre          |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but (as
per usual SBML principles) their symbols represent their values in
concentration units where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)



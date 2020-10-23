(* 

category:      Test
synopsis:      Basic single forward reaction with two species and a delayed event.
componentTags: Compartment, Species, Reaction, Parameter, EventWithDelay
testTags:      Amount, EventIsNotPersistent
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "C".  There are three
species named S1, S2 and S3 and one parameter named k1.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |]

The model contains one event that assigns value to species S3 defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $S1 <= 0.00005 && S1 >= 0.00004$ | $3$   | $S3 = 2.5 \x 10^-4$    |]
 
The persistent attribute of the Trigger is set to 'false'.  Thus, once the event has fired,
if the trigger value reverts to 'false' before the event is executed, then execution should
not take place.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$1.5 \x 10^-4$  |mole           |
|Initial amount of S2                |$1.5 \x 10^-4$  |mole           |
|Initial amount of S2                |$1.5 \x 10^-4$  |mole           |
|Value of parameter k1               |$1$              |second^-1^     |
|Volume of compartment "C" |$1$              |litre          |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)



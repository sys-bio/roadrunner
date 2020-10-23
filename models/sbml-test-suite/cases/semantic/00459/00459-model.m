(* 

category:      Test
synopsis:      Model using parameters and rules only with
               and one event that assigns value to two parameters, subject to a delay.
componentTags: Parameter, RateRule, EventWithDelay 
testTags:      NonConstantParameter, EventUsesTriggerTimeValues
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called S1, S2 and S3 and two constant
parameters called k1 and k2.  (Note that indeed S1, S2 and S3 are parameters and 
not species in this model.)

The model contains three rules defined as:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S1  | $k2 * S3 - k1 * S1 * S2$  |
 | Rate | S2  | $k2 * S3 - k1 * S1 * S2$  |
 | Rate | S3  | $k1 * S1 * S2 - k2 * S3$  |]

The model contains one event that assigns values to both parameters S1 and S2:

[{width:30em,margin: 1em auto}| | *Trigger*   | *Delay* | *Assignments* |
 | Event1                      | $S1 < 0.75$ | $2$     | $S2 = 1.5$    |
 |                             |             |         | $S1 = S2$     |]
 
Note that the event assignments should happen simultaneously, not
sequentially; i.e., the value of S2 assigned to S1 is the value at the
point at which the event was triggered.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$1.0$  |any                      |
|Initial amount of S2  |$2.0$  |same as S1                      |
|Initial amount of S3  |$1.0$  |same as S1                      |
|Value of parameter k1 |$0.75$           |(unit of S1)^-1^ second^-1^ |
|Value of parameter k2 |$0.25$           |second^-1^                |]


*)

newcase[ "00459" ];

addParameter[ S1, value -> 1.0, constant -> False];
addParameter[ S2, value -> 2.0, constant -> False];
addParameter[ S3, value -> 1.0, constant -> False];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->RateRule, variable -> S1, math -> k2 * S3 - k1 * S1 * S2];
addRule[ type->RateRule, variable -> S2, math -> k2 * S3 - k1 * S1 * S2];
addRule[ type->RateRule, variable -> S3, math -> k1 * S1 * S2 - k2 * S3];
addEvent[ trigger -> S1 < 0.75, delay->2, eventAssignment -> {S2->1.5, S1->S2} ];

makemodel[]


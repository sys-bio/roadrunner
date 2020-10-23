(* 

category:      Test
synopsis:      Basic two reactions with four species in a compartment whose 
               volume is varying.
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one varying compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and p1.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]

The model contains one rule which defines the rate at which the volume of
the compartment is changing:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | C | $-p1 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$          1.0$ |mole                      |
|Initial amount of S2                |$          1.5$ |mole                      |
|Initial amount of S3                |$          2.0$ |mole                      |
|Initial amount of S4                |$          1.0$ |mole                      |
|Value of parameter k1               |$         0.75$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$         0.25$ |litre mole^-1^ second^-1^ |
|Value of parameter p1               |$          0.1$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00053" ];

addCompartment[ C, constant->False, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 1.5];
addSpecies[ S3, initialAmount -> 2.0];
addSpecies[ S4, initialAmount -> 1.0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.1 ];
addRule[ type->RateRule, variable -> C, math -> -p1 * C];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];

makemodel[]


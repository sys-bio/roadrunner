(* 

category:      Test
synopsis:      One reactions and two rate rules with four species in a compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and two parameters called k1 and k2.  The
model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |]

Reaction S1 -> S2 defines one local parameter k1 which has a
scope local to the defining reaction and is different from the global parameter k1
used in the RateRule.

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3 | $k1  *  0.5$  |
 | Rate | S4 | $-k2  *  0.5$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.5$ |mole                      |
|Initial amount of S2                |$          2.0$ |mole                      |
|Initial amount of S3                |$          1.5$ |mole                      |
|Initial amount of S4                |$            4$ |mole                      |
|Value of parameter k1               |$         0.75$ |second^-1^ |
|Value of parameter k2               |$         0.25$ |second^-1^ |
|Value of local parameter k1               |$0.9$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00732" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.5];
addSpecies[ S4, initialAmount -> 4];
addParameter[ k1, value -> .75 ];
addParameter[ k2, value -> .25 ];
addRule[ type->RateRule, variable -> S3, math -> k1  *  0.5];
addRule[ type->RateRule, variable -> S4, math -> -k2  *  0.5];
addReaction[S1 -> S2,  reversible -> False,
	     kineticLaw -> k1 * S1 * C, parameters -> {k1 -> 0.9} ];

makemodel[]


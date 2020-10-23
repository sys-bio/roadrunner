(* 

category:      Test
synopsis:      Two reactions and a rate rule that applies a functionDefinition
               with four species in a compartment. 
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, RateRule 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and one parameter called k.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k * S3 * C$  |]

Each reaction defines one local parameter k which has a
scope local to the defining reaction and is different from the global parameter k
used in the RateRule.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S4 | $multiply(k2, 4 \x 10^-6$  |]

The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S2                |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S3                |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S4                |$   1 \x 10^-7$ |mole                      |
|Value of parameter k               |$         0.25$ |second^-1^ |
|Value of local parameter k (R1)               |$ 0.75 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of local parameter k (R2)              |$         0.3$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00733" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-6];
addSpecies[ S2, initialAmount -> 2.0 10^-6];
addSpecies[ S3, initialAmount -> 1.5 10^-6];
addSpecies[ S4, initialAmount -> 1 10^-7 ];
addParameter[ k, value -> .25 ];
addRule[ type->RateRule, variable -> S4, math -> multiply[k, 4 10^-6]];
addReaction[S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 0.75 10^6} ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * C, parameters -> {k -> 0.3} ];

makemodel[]


(* 

category:      Test
synopsis:      Two reactions with four species in one
compartment using an assignmentRule to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S and one parameter called k.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k * S3 * C$  |]

Reaction S1 + S2 -> S3 defines one local parameter k.  Reaction S3 -> S1 + S2 also
defines one (different) local parameter k.  Both of these parameters have a
scope local to the defining reaction.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S4 | $k * S2$  |]

Note that in this case the initial value of species S4 has not been declared and must be 
calculated using the assignmentRule.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$    1 \x 10^-5 |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-5$ |mole                      |
|Initial amount of S3                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S4                |$   undeclared$ |mole                      |
|Value of parameter k                |$          1.5$ |dimensionless |
|Value of local parameter k          |$  1.5 \x 10^4$ |litre mole^-1^ second^-1^ |
|Value of local parameter k          |$            5$ |second^-1^ |
|Volume of compartment C             |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00612" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1 10^-5 ];
addSpecies[ S2, initialAmount -> 1.5 10^-5];
addSpecies[ S3, initialAmount -> 1 10^-5];
addSpecies[ S4];
addParameter[ k, value -> 1.5 ];
addRule[ type->AssignmentRule, variable -> S4, math ->k * S2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 1.5 10^4} ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * C, parameters -> {k -> 5} ];

makemodel[]


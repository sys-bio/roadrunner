(* 

category:      Test
synopsis:      Basic two reactions with four species in one compartment with
an algebraic rule used to determine rate of change of one species.
componentTags: Compartment, Species, Reaction, AlgebraicRule 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k * S3 * C$     |]


Reaction S1 + S2 -> S3 defines one local parameter k.  Reaction S3 -> S1 +
S2 defines another local parameter k.  Note that these parameters have a
scope local to the defining reaction.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $S4 - S1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial amount of S1                |$1.0 \x 10^-2$  |mole                      |
|Initial amount of S2                |$2.0 \x 10^-2$  |mole                      |
|Initial amount of S3                |$1.0 \x 10^-2$  |mole                      |
|Initial amount of S4                |$1.0 \x 10^-2$  |mole                      |
|Value of local parameter k          |$0.75$           |litre mole^-1^ second^-1^ |
|Value of local parameter k          |$0.25$           |second^-1^                |
|Volume of compartment C             |$1$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00566" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-2];
addSpecies[ S2, initialAmount -> 2.0 10^-2];
addSpecies[ S3, initialAmount -> 1.0 10^-2];
addSpecies[ S4, initialAmount -> 1.0 10^-2];
addRule[ type->AlgebraicRule, math -> S4 - S1];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 0.75}  ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * C, parameters -> {k -> 0.25}  ];

makemodel[]


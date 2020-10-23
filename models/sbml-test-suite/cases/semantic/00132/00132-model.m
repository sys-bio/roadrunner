(* 

category:      Test
synopsis:      Basic two reactions using functionDefinitions with three species 
               and parameters local to reactions.
componentTags: Compartment, Species, Reaction, FunctionDefinition 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $multiply(k,S1) * compartment,$  |
| S2 -> S3 | $k * S2 * compartment,$  |]

Reaction S1 -> S2 defines one local parameter k.  Reaction S2 -> S3 defines
another (different) local parameter k.


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$3.0 \x 10^-3$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Initial amount of S3                |$            0$ |mole                      |
|Value of local parameter k          |$            1$ |second^-1^ |
|Value of local parameter k          |$            2$ |second^-1^ |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00132" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 3.0 10^-3];
addSpecies[ S2, initialAmount -> 0];
addSpecies[ S3, initialAmount -> 0];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> multiply[k,S1] * compartment, parameters -> {k -> 1} ];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k * S2 * compartment, parameters -> {k -> 2} ];

makemodel[]


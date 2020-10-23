(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
compartment using an assignmentRule to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and one parameters called k.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k * S$  |]

Reaction S1 -> S2 defines one local parameter k which has a
scope local to the defining reaction.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $k * S2$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*      |*Units*  |
|Initial amount of S1                |$1$          |mole                      |
|Initial amount of S2                |$0$          |mole                      |
|Initial amount of S3                |$1$          |mole                      |
|Value of parameter k                |$0.75$       |dimensionless |
|Value of local parameter k          |$5$          |second^-1^ |
|Volume of compartment C             |$1$          |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00611" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1 ];
addSpecies[ S2, initialAmount -> 0 ];
addSpecies[ S3, initialAmount -> 1];
addParameter[ k, value -> 0.75 ];
addRule[ type->AssignmentRule, variable -> S3, math ->k * S2];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k * S1, parameters -> {k -> 5}];

makemodel[]


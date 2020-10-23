(* 

category:      Test
synopsis:      Two reversible reactions with local parameters.
componentTags: Compartment, Species, Reaction 
testTags:      Amount, ReversibleReaction, LocalParameters
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a copy of model 831, with the products and reactants reversed, 
and a negative kinetic law.  This gives the same results, but during the simulation, 
the kinetic law actually goes negative (a requirement for reversibe reactions).  
The reaction listed below is therfore equivalent to the one in the file, but reversed from it.

The model contains one compartment called C.  There are two species called 
S1 and S2.  The model contains  two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 <-> S2 | $(kf * S1 - kr * S2) * C$  |
| S3 <-> S4 | $(kf * S3 - kr * S4) * C$  |]

Both reactions define two local parameters kf and kr which have a
scope local to the defining reaction.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0$ |mole                      |
|Value of local parameter kf (R1)       |$0.8$          |second^-1^ |
|Value of local parameter kr (R1)      |$0.06$          |second^-1^ |
|Value of local parameter kf (R1)      |$0.9$          |second^-1^ |
|Value of local parameter kr (R2)      |$0.075$          |second^-1^ |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00831" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0];
addSpecies[ S3, initialAmount -> 1.5];
addSpecies[ S4, initialAmount -> 0.5];
addReaction[ S1 -> S2,
	     kineticLaw -> (kf * S1 - kr * S2) * C , parameters -> {kf -> 0.8, kr -> 0.06}];
addReaction[ S3 -> S4,
	     kineticLaw -> (kf * S3 - kr * S4) * C , parameters -> {kf -> 0.9, kr -> 0.075}];

makemodel[]


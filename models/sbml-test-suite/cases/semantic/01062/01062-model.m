(* 

category:      Test
synopsis:      Single reversible reaction.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, ReversibleReaction, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

 Note:  This test is a copy of model 816, with the products and reactants reversed, and a negative kinetic law.  This gives the same results, but during the simulation, the kinetic law actually goes negative (a requirement for reversibe reactions).  The reaction listed below is therfore equivalent to the one in the file, but reversed from it.

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and two parameters called kf and kr.  The model contains  one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 <-> S3 | $(kf * S1 * S2 - kr * S3) * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial concentration of S1        |$1.0$ |mole litre^-1^                      |
|Initial concentration of S2        |$0.5$ |mole litre^-1^                      |
|Initial concentration of S3        |$0$ |mole litre^-1^                      |
|Value of parameter kf       |$1.1$          |litre mole^-1^ second^-1^ |
|Value of parameter kr       |$0.09$          |second^-1^ |
|Volume of compartment C     |$2.3$             |litre                  |]

*)

newcase[ "00816" ];

addCompartment[ C, size -> 2.3 ];
addSpecies[ S1, initialConcentration -> 1.0];
addSpecies[ S2, initialConcentration -> 0.5];
addSpecies[ S3, initialConcentration -> 0.0];
addParameter[ kf, value -> 1.1 ];
addParameter[ kr, value -> 0.09 ];
addReaction[ S1 + S2 -> S3,
	     kineticLaw -> (kf * S1 * S2 - kr * S3) * C ];

makemodel[]


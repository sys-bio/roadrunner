(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment, nonzero initial concentrations, and identical reaction rates.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two species called
S1 and S2 and two parameters called k1 and k2.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |
| S2 -> S1 | $k2 * S2 * C$  |]

The rate constants are identical in both directions, leading to a system
where the species concentrations remain the same.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*        |
|Initial concentration of S1 |$1.5 \x 10^-3$ |mole litre^-1^ |
|Initial concentration of S2 |$1.5 \x 10^-3$ |mole litre^-1^ |
|Value of parameter k1       |$0.5$           |second^-1^     |
|Value of parameter k2       |$0.5$           |second^-1^     |
|Volume of compartment C     |$1$             |litre          |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00464" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 1.5 10^-3 ];
addSpecies[ S2, initialConcentration -> 1.5 10^-3 ];
addParameter[ k1, value -> 0.5 ];
addParameter[ k2, value -> 0.5 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];
addReaction[ S2 -> S1, reversible -> False,
	     kineticLaw -> k2 * S2 * C ];

makemodel[]


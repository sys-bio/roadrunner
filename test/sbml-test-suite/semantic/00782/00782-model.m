(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of the compartment.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Concentration, NonUnityCompartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two species called 
S1 and S2 and two parameters called k1 and k2.  The model contains one 
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k1 * S1$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | C | $k2 / 10$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for compartment C is
consistent with the value returned by the InitialAssignment.


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*  |*Units*  |
|Initial concentration of S1        |$1$        |mole litre^-1^                      |
|Initial concentration of S2        |$1.5$      |mole litre^-1^                      |
|Value of parameter k1       |$0.5$      |second^-1^ |
|Value of parameter k2       |$50$       |litre |
|Volume of compartment C     |$5$        |litre                     |]

*)

newcase[ "00782" ];

addCompartment[ C, size -> 5, constant -> False];
addSpecies[ S1, initialConcentration -> 1 ];
addSpecies[ S2, initialConcentration -> 1.5 ];
addParameter[ k1, value -> 0.5];
addParameter[ k2, value -> 50 ];
addInitialAssignment[ C, math -> k2/10];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k1 * S1];

makemodel[]


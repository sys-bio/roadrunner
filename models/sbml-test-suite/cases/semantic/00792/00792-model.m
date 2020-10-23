(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of one parameter.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Concentration, InitialValueReassigned
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
 | k1 | $k2 / 100$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the parameter k1 has not been
explicitly declared and must be calculated using the InitialAssignment.


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*  |*Units*  |
|Initial concentration of S1    |$1$              |mole litre^-1^   |
|Initial concentration of S2    |$1.5$            |mole litre^-1^   |
|Value of parameter k1   |$undeclared$     |second^-1^ |
|Value of parameter k2   |$50$             |second^-1^ |
|Volume of compartment C |$1$              |litre      |]

*)

newcase[ "00792" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialConcentration -> 1 ];
addSpecies[ S2, initialConcentration -> 1.5 ];
addParameter[ k1, constant->False ];
addParameter[ k2, value -> 50 ];
addInitialAssignment[ k1, math -> k2/100];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k1 * S1];

makemodel[]


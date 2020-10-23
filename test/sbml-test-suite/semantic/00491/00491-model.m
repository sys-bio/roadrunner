(* 

category:      Test
synopsis:      Basic two reactions with three species in a 0D compartment using 
initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, 0D-Compartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and three parameters called k1, k2 and p1.  Compartment C is 
0-dimensional.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 $  |
| S3 -> S1 + S2 | $k2 * S3$  |]

  The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the species S1 has not been
explicitly declared and must be calculated using the InitialAssignment.

The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*        |*Units*  |
|Initial amount of S1        |$undeclared$     |mole                      |
|Initial amount of S2        |$2.0 \x 10^-1$  |mole                      |
|Initial amount of S3        |$1.0 \x 10^-1$  |mole                      |
|Value of parameter k1       |$0.75$           |mole^-1^ second^-1^ |
|Value of parameter k2       |$0.25$           |second^-1^                |
|Value of parameter p1       |$1.25 \x 10^-2$ |mole                |]

In this example, the compartment has its spatialDimensions attribute set to
zero, i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00491" ];

addCompartment[ C, spatialDimensions -> 0 ];
addSpecies[ S1];
addSpecies[ S2, initialAmount -> 2.0 10^-1];
addSpecies[ S3, initialAmount -> 1.0 10^-1];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.125 10^-1 ];
addInitialAssignment[ S1, math -> p1*2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]


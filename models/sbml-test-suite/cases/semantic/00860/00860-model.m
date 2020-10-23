(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one compartment
using csymbol time within a math expression.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime 
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two
species called S1 and S2 and one parameter called k1.  The model contains
one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C * t$ |]
where the symbol 't' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial concentration of S1                |$1.5 \x 10^-3$  |mole litre^-1^           |
|Initial concentration of S2                |$0$              |mole litre^-1^           |
|Value of parameter k1               |$1$              |second^-2^     |
|Volume of compartment C |$0.9$              |litre          |]

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00860" ];

addCompartment[ C, size -> 0.9 ];
addSpecies[ S1, initialConcentration -> 1.5 10^-3 ];
addSpecies[ S2, initialConcentration -> 0 ];
addParameter[ k1, value -> 1 ];
addReaction[ S1 -> S2, reversible -> False, 
 kineticLaw -> k1 * S1 * C *\[LeftAngleBracket]t, "time"\[RightAngleBracket]];

makemodel[]


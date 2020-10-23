(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment of non-unity size using a function to apply 
the kinetic law.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two
species called S1 and S2 and one parameter called k1.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * multiply(k1,S1)$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$1.5 \x 10^-3$ |mole litre^-1^                      |
|Initial concentration of S2   |$1.5 \x 10^-3$ |mole litre^-1^                      |
|Value of parameter k1         |$            1$ |second^-1^ |
|Volume of compartment C       |$          2.3$ |litre                     |]

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00604" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 2.3 ];
addSpecies[ S1, initialConcentration -> 1.5 10^-3 ];
addSpecies[ S2, initialConcentration -> 1.5 10^-3 ];
addParameter[ k1, value -> 1 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * multiply[k1,S1] ];

makemodel[]


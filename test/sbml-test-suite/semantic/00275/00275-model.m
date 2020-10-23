(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using MathML in the rate equation.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Amount
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named p1 and p2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $compartment * S1 * calculate(p1, p2)$  |]

The model contains one functionDefinition, which is used within the reaction, defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | calculate | x, y | $Piecewise({{x, True}}, y)$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter p1               |$          1.5$ |second^-1^ |
|Value of parameter p2               |$         0.05$ |second^-1^ |
|Volume of compartment "compartment" |$           1.$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00275" ];

addFunction[ calculate, arguments -> {x, y}, math -> Piecewise[{{x, True}}, y] ];
addCompartment[ compartment, size -> 1.0];
addSpecies[ S1, initialAmount -> 1];
addSpecies[ S2, initialAmount -> 0];
addParameter[ p1, value -> 1.5 ];
addParameter[ p2, value -> 0.05 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> compartment * S1 * calculate[p1,p2] ];

makemodel[]


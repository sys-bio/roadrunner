(* 

category:      Test
synopsis:      Basic two reactions with four species in one 0D compartment
               and one function definition.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Amount, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

This model contains one 0-dimensional compartment called "compartment".
There are four species, S1, S2, S3 and S4, and two parameters, k1 and k2.

The model contains a function called "multiply" that takes two numerical
arguments and multiplies them:

[{width:25em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
| multiply | x, y | $x  *  y$ |]

The model contains two reactions defined as:

[{width:25em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1  +  S2 -> S3  +  S4 | $multiply(k1, multiply(S1, S2))$ |
| S3  +  S4 -> S1  +  S2 | $k2  *  S3  *  S4$                       |]

The initial conditions are as follows:

[{width:25em,margin: 1em auto}|      |*Value*            |*Units*              |
|Initial amount of S1  |$1.0 \x 10^-4$ |mole                |
|Initial amount of S2  |$1.0 \x 10^-4$ |mole                |
|Initial amount of S3  |$2.0 \x 10^-4$ |mole                |
|Initial amount of S4  |$1.0 \x 10^-4$ |mole                |
|Value of parameter k1 |$0.75 \x 10^4$ |mole^-1^ second^-1^ |
|Value of parameter k2 |$0.25 \x 10^4$ |mole^-1^ second^-1^ |]

In this example, the compartment has its spatialDimensions attribute set to
a value of zero; i.e., the compartment represents a point, and therefore
cannot have a value for the %size% or %units% attributes.  The quantities
of species located in the compartment must be treated as amounts and not
concentrations.

*)

newcase[ "00103" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, spatialDimensions-> 0 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4];
addSpecies[ S2, initialAmount -> 1.0 10^-4];
addSpecies[ S3, initialAmount -> 2.0 10^-4];
addSpecies[ S4, initialAmount -> 1.0 10^-4];
addParameter[ k1, value -> 0.75 10^4 ];
addParameter[ k2, value -> 0.25 10^4 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> multiply[k1,multiply[S1,S2]] ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 ];

makemodel[]


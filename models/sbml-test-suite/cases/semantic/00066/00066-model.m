(* 

category:      Test
synopsis:      Rate rule used to determine parameter value which is used 
               in a reaction.
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, NonConstantParameter
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species called S1, S2, S3 and S4; one constant parameter called k2 and one
varying parameter called k2.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * compartment$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * compartment$  |]

The model contains one rule which defines that rate at which parameter k1
varies:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | k1 | $10^6$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S3                |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4                |$ 0.5 \x 10^-6$ |mole                      |
|Value of parameter k1               |$    1 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$  0.3 \x 10^6$ |litre mole^-1^ second^-1^ |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00066" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 1.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 0.5 10^-6];
addParameter[ k1, value -> 1 10^6, constant->False ];
addParameter[ k2, value -> 0.3 10^6 ];
addRule[ type->RateRule, variable -> k1, math -> 10^6];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]


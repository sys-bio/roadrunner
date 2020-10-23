(* 

category:      Test
synopsis:      Two reactions with four species in one
compartment using an assignmentRule and a rateRule to vary species.
componentTags: Compartment, Species, Reaction, Parameter, RateRule, AssignmentRule, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and four parameters called k1, k2, k3 and p1.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S3 | $k1 * S1 * C$  |
| S3 -> S1 | $k2 * S3 * C$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S4 | $k3 * S1$  |
 | Rate       | S2 | $k2 * S3-k1 * S1$  |]

Note that in this case the initial value declared for species S4 is inconsistent
with the value calculated by the assignmentRule.  The calculated value should be used.  

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 / 2000$  |]

Note: InitialAssignments override any declared initial values.  In this case the 
value of species S1 is not declared by the model definition.  The calculated value should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$ undeclared$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-5$ |mole                      |
|Initial amount of S3                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S4                |$            5$ |mole                      |
|Value of parameter k1               |$        0.015$ |second^-1^ |
|Value of parameter k2               |$          0.5$ |second^-1^ |
|Value of parameter k3               |$          1.5$ |dimensionless |
|Value of parameter p1               |$   2 \x 10^-2$ |mole litre^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00645" ];

addCompartment[ C, size -> 1];
addSpecies[ S1];
addSpecies[ S2, initialAmount -> 1.5 10^-5];
addSpecies[ S3, initialAmount -> 1 10^-5];
addSpecies[ S4, initialAmount -> 5];
addParameter[ k1, value -> .015];
addParameter[ k2, value -> .5 ];
addParameter[ k3, value -> 1.5 ];
addParameter[ p1, value -> 2.0 10^-2 ];
addInitialAssignment[ S1, math -> p1/2000];
addRule[ type->AssignmentRule, variable -> S4, math ->k3 * S1];
addRule[ type->RateRule, variable -> S2, math ->k2 * S3-k1 * S1];
addReaction[ S1 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];
addReaction[ S3 -> S1, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]


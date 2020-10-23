(* 

category:      Test
synopsis:      One reactions and two rate rules with four species in a compartment. 
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, RateRule 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 729.

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and four parameters called k1, k2, p1 and p2.  The
model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> (4 * p2)S2 | $k1 * S1 * C$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3 | $k2  *  S4$  |
 | Rate | S4 | $-k2  *  S4$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 / 0.5$  |]

Note: InitialAssignments override any declared initial values.  In this case the 
value of species S1 is not declared by the model definition.  The calculated value should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$ undeclared$ |mole                      |
|Initial amount of S2                |$          2.0$ |mole                      |
|Initial amount of S3                |$          1.5$ |mole                      |
|Initial amount of S4                |$            4$ |mole                      |
|Value of parameter k1               |$         0.75$ |second^-1^ |
|Value of parameter k2               |$         0.5$ |second^-1^ |
|Value of parameter p1                |$         0.75$ |mole litre^-1^ |
|Value of parameter p2               |$0.25$ |dimensionless |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00729" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1 ];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.5];
addSpecies[ S4, initialAmount -> 4];
addParameter[ k1, value -> .75 ];
addParameter[ k2, value -> .5 ];
addParameter[ p1, value -> .75 ];
addParameter[ p2, value -> 0.25];
addInitialAssignment[ S1, math -> p1/0.5];
addRule[ type->RateRule, variable -> S3, math -> k2  *  S4];
addRule[ type->RateRule, variable -> S4, math -> -k2  *  S4];
addReaction[reactants->{S1}, products->{S2}, productStoichiometry->{4 * p2}, 
       reversible -> False,
	     kineticLaw -> k1 * S1 * C ];

makemodel[]


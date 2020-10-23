(* 

category:      Test
synopsis:      Basic single forward reaction with two species in a compartment
               whose volume is varying.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule, RateRule 
testTags:      Amount, NonConstantCompartment, NonUnityCompartment, NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one varying compartment called C.  There are two
species called S1 and S2 and three parameters called k1, p1 and p2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |]

The model contains two rules. The assignmentRule assigns value to C
and the rateRule determines the rate at which p2 is varying:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | C       | $p1 * p2$  |
 | Rate       | p2      | $0.1       |]

In this case the initial value declared for C is consistent
with that calculated by the assignmentRule.  Note that since this
assignmentRule must always remain true, it should be considered during
simulation.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.5$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter k1               |$          0.9$ |second^-1^ |
|Value of parameter p1               |$          0.1$ |litre mole^-1^ |
|Value of parameter p2               |$          1.5$ |mole second^-1^ |
|Volume of compartment C |$         0.15$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00310" ];

addCompartment[ C, size->0.15, constant->False ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 0.9 ];
addParameter[ p1, value -> 0.1 ];
addParameter[ p2, value -> 1.5, constant -> False];
addRule[ type->AssignmentRule, variable -> C, math -> p1 * p2];
addRule[ type->RateRule, variable -> p2, math -> 0.1];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];

makemodel[]


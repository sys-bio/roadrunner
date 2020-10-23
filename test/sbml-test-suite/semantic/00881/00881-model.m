(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
compartment using an assignmentRule with csymbol time to vary one species.
componentTags: Compartment, Species, CSymbolTime, Reaction, Parameter, AssignmentRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S and two parameters called k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k2 * S1$  |]

The model contains one rule which assigns value to species S3:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $k1 * S2 * t$  |]
where the symbol 't' denotes the current simulation time.

In this case the initial value for species S3 is inconsistent with the
value calculated by the assignmentRule; the calculated value should be
used.  Note that since this assignmentRule must always remain
true, it should be considered during simulation.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$1.5 \x 10^-15$ |mole                      |
|Initial amount of S3                |$   1$ |mole                      |
|Value of parameter k1               |$         0.75$ |second^-1^ |
|Value of parameter k2               |$          2.5$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00881" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1.5 ];
addSpecies[ S2, initialAmount -> 1.5 10^-15 ];
addSpecies[ S3, initialAmount->1 ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value ->2.5 ];
addRule[ type->AssignmentRule, variable -> S3, math ->k1 * S2 * \[LeftAngleBracket]t, "time"\[RightAngleBracket]];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k2 * S1];

makemodel[]


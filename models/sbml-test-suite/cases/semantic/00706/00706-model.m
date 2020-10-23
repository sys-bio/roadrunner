(* 

category:      Test
synopsis:      Basic reaction with two species in one
compartment using an assignmentRule to assign value to a parameter.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule, InitialAssignment 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two
species called S1 and S2 and one parameter called k.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k * S1$  |]

Reaction S1 -> S2 defines one local parameter k which has a
scope local to the defining reaction and is different from the global parameter k
used in the initialAssignment.

The model contains one rule which assigns value to parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | k | $0.75$  |]

In this case there is no initial value declared for parameter k.  Thus the
value must be calculated using the assignmentRule.

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $k / 7.5$  |]

Note: InitialAssignments override any declared initial values.  The initial
value of species S1 is inconsisent with the value declared by the model definition.  
The value calculated by the initialAssignment should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$10$    |mole                      |
|Initial amount of S2                |$         0.15$ |mole                      |
|Value of parameter k               |$   undeclared$ |second^-1^ |
|Value of local parameter k          |$   0.75$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00706" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount -> 10 ];
addSpecies[ S2, initialAmount -> .15 ];
addParameter[ k, constant -> False ];
addInitialAssignment[ S1, math -> k/7.5];
addRule[ type->AssignmentRule, variable -> k, math -> 0.75];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k * S1, parameters -> {k -> 0.75}];

makemodel[]


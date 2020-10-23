(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
compartment using an assignmentRule with a functionDefinition to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, AssignmentRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $compartment * k2 * S$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $multiply(k1, S1)$  |]

The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]

Note that in this case the initial value of the species S3 is not explicitly
declared and must be calculated by the AssignmentRule. 

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$          0.5$ |mole                      |
|Initial amount of S3                |$   undeclared$ |mole                      |
|Value of parameter k1               |$          1.5$ |dimensionless |
|Value of parameter k2               |$            5$ |second^-1^ |
|Volume of compartment "compartment" |$             $ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00078" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, size -> 1];
addSpecies[ S1, initialAmount->1 ];
addSpecies[ S2, initialAmount -> 0.5 ];
addSpecies[ S3 ];
addParameter[ k1, value -> 1.5 ];
addParameter[ k2, value -> 5 ];
addRule[ type->AssignmentRule, variable -> S3, math ->multiply[k1,S2]];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> compartment * k2 * S1];

makemodel[]


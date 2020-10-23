(* 

category:      Test
synopsis:      Basic reaction with two species in one
compartment using an assignmentRule to assign value to a parameter.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule, InitialAssignment, FunctionDefinition 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two
species called S1 and S2 and three parameters called k1, p1 and p2.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k1 * S1$  |]

The model contains one rule which assigns value to parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | k1 | $multiply(p1, p2)$  |]

The assignmentRule applies the functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y      | $x * y$   |]
 
In this case there is no initial value declared for parameter k1.  Thus the
value must be calculated using the assignmentRule.

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $k1 / 7.5$  |]

Note: InitialAssignments override any declared initial values.  The initial
value of species S1 is inconsisent with the value declared by the model definition.  
The value calculated by the initialAssignment should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$10$    |mole                      |
|Initial amount of S2                |$         0.15$ |mole                      |
|Value of parameter k1               |$   undeclared$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00632" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount -> 10];
addSpecies[ S2, initialAmount -> .15 ];
addParameter[ k1, constant -> False ];
addParameter[ p1, value -> 75 ];
addParameter[ p2, value -> 0.01 ];
addInitialAssignment[ S1, math -> k1/7.5];
addRule[ type->AssignmentRule, variable -> k1, math -> multiply[p1, p2]];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k1 * S1];

makemodel[]


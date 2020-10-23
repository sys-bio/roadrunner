(* 

category:      Test
synopsis:      Basic rule that assigns value to a species.
componentTags: Compartment, Species, AssignmentRule 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There is one
species named S1.  The model does not contain any reactions.
  
The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| assignmentRule | S1 | $3 + 4$  |]

The rule assigns a value to species S1 which is consistent with the value
attributed to the species by the model definition.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            7$ |mole                      |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00029" ];

addCompartment[ compartment, size -> 1.0];
addSpecies[ S1, initialAmount -> 7 ];
addRule[ type -> assignmentRule, variable -> S1, math -> 3 + 4];

makemodel[]


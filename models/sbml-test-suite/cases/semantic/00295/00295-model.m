(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
compartment using an AssignmentRule to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      Amount, HasOnlySubstanceUnits, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k2 * S1$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3       | $k1 * S2$  |]

In this case there is no initial value declared for species S3.  Thus the
initial value must be calculated by the assignmentRule.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1                |$1.0 \x 10^-1$ |mole                      |
|Initial amount of S2                |$1.5 \x 10^-1$ |mole                      |
|Initial amount of S3                |$   undeclared$ |mole                      |
|Value of parameter k1               |$         1.05$ |dimensionless             |
|Value of parameter k2               |$         1.15$ |second^-1^                |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species have been declared as having substance units only. Thus, they 
must be treated as amounts where they appear in expressions.

*)

newcase[ "00295" ];

addCompartment[ compartment, size -> 1];
addSpecies[ S1, initialAmount-> 1 10^-1, hasOnlySubstanceUnits->True ];
addSpecies[ S2, initialAmount -> 1.5 10^-1, hasOnlySubstanceUnits->True ];
addSpecies[ S3, hasOnlySubstanceUnits->True  ];
addParameter[ k1, value -> 1.05 ];
addParameter[ k2, value -> 1.15 ];
addRule[ type->AssignmentRule, variable -> S3, math ->k1 * S2];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k2 * S1];

makemodel[]


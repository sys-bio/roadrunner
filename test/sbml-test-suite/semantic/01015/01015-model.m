(* 

category:      Test
synopsis:      Two reactions with four species in one
               size-10 compartment using an assignmentRule to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      Amount, HasOnlySubstanceUnits, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a direct copy of test 00296, but with a size-10 compartment, to test the 'hasOnlySubstanceUnits' attribute.  The model should have exactly the same results, as concentration should have never been used to calculate anything.
 
The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and three parameters named k1, k2 and k3.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2$  |
| S3 -> S1 + S2 | $k2 * S3$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S4 | $k3 * S2$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-5$ |mole                      |
|Initial amount of S3                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S4                |$2.25 \x 10^-5$ |mole                      |
|Value of parameter k1               |$  1.5 \x 10^5$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$           50$ |second^-1^ |
|Value of parameter k3               |$          1.5$ |dimensionless |
|Volume of compartment "compartment" |$            10$ |litre                     |]

The species have been declared as having substance units only. Thus, they 
must be treated as amounts where they appear in expressions.

*)

newcase[ "00296" ];

addCompartment[ compartment, size -> 10];
addSpecies[ S1, initialAmount->1 10^-5 , hasOnlySubstanceUnits->True ];
addSpecies[ S2, initialAmount -> 1.5 10^-5, hasOnlySubstanceUnits->True ];
addSpecies[ S3, initialAmount -> 1 10^-5, hasOnlySubstanceUnits->True ];
addSpecies[ S4, initialAmount -> 2.25 10^-5, hasOnlySubstanceUnits->True ];
addParameter[ k1, value -> 1.5 10^5];
addParameter[ k2, value -> 50 ];
addParameter[ k3, value -> 1.5 ];
addRule[ type->AssignmentRule, variable -> S4, math ->k3 * S2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]


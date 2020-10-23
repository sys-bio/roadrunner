(* 

category:      Test
synopsis:      Basic reaction with two species in a 2-dimensional
size-10 compartment, where the species have only substance units.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, HasOnlySubstanceUnits, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a direct copy of test 00209, but with a size-10 compartment, to test the 'hasOnlySubstanceUnits' attribute.  The model should have exactly the same results, as concentration should have never been used to calculate anything.
 
The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  Species S1, S2 and S3 are
declared to have only substance units.  The model contains two reactions
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2$  |
| S3 -> S1 + S2 | $k2 * S3$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1              |$1.0 \x 10^-1$ |mole                      |
|Initial amount of S2              |$2.0 \x 10^-1$ |mole                      |
|Initial amount of S3              |$1.0 \x 10^-1$ |mole                      |
|Value of parameter k1             |$           75$ |mole^-1^ second^-1^ |
|Value of parameter k2             |$           25$ |second^-1^ |
|Area of compartment "compartment" |$            10$ |metre^2^                  |]

The species have been declared as having substance units only.  Thus, they
must be treated as amounts where they appear in expressions.

*)

newcase[ "00209" ];

addCompartment[ compartment, spatialDimensions-> 2, size -> 10 ];
addSpecies[ S1, initialAmount -> 1.0 10^-1, hasOnlySubstanceUnits->True ];
addSpecies[ S2, initialAmount -> 2.0 10^-1, hasOnlySubstanceUnits->True ];
addSpecies[ S3, initialAmount -> 1.0 10^-1, hasOnlySubstanceUnits->True ];
addParameter[ k1, value -> 75 ];
addParameter[ k2, value -> 25 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]


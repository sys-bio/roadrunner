(* 

category:      Test
synopsis:      Basic two reactions with four species in a 2-dimensional
compartment, where the species have only substance units.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, HasOnlySubstanceUnits, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a direct copy of test 00210, but with a size-10 compartment, to test the 'hasOnlySubstanceUnits' attribute.  The model should have exactly the same results, as concentration should have never been used to calculate anything.
 
The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  All four species are declared
to have only substance units.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1              |$1.0 \x 10^-3$ |mole                      |
|Initial amount of S2              |$1.5 \x 10^-3$ |mole                      |
|Initial amount of S3              |$2.0 \x 10^-3$ |mole                      |
|Initial amount of S4              |$1.0 \x 10^-3$ |mole                      |
|Value of parameter k1             |$ 7.5 \x 10^3$ |mole^-1^ second^-1^ |
|Value of parameter k2             |$0.25 \x 10^3$ |mole^-1^ second^-1^ |
|Area of compartment "compartment" |$            10$ |metre^2^                  |]

The species have been declared as having substance units only.  Thus, they 
must be treated as amounts where they appear in expressions.

*)

newcase[ "00210" ];

addCompartment[ compartment, spatialDimensions-> 2, size -> 10 ];
addSpecies[ S1, initialAmount -> 1.0 10^-3, hasOnlySubstanceUnits->True];
addSpecies[ S2, initialAmount -> 1.5 10^-3, hasOnlySubstanceUnits->True];
addSpecies[ S3, initialAmount -> 2.0 10^-3, hasOnlySubstanceUnits->True];
addSpecies[ S4, initialAmount -> 1.0 10^-3, hasOnlySubstanceUnits->True];
addParameter[ k1, value -> 7.5 10^3 ];
addParameter[ k2, value -> 0.25 10^3 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 ];

makemodel[]


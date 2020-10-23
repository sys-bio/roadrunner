(* 

category:      Test
synopsis:      One reactions and two rate rules with four species in a size-10 compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, HasOnlySubstanceUnits, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

Note:  This test is a direct copy of test 00333, but with a size-10 compartment, to test the 'hasOnlySubstanceUnits' attribute.  The model should have exactly the same results, as concentration should have never been used to calculate anything.
 
The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.  The
model contains one reaction defined as:

[{width:25em,=}|  *Reaction*  |  *Rate*               |
|              S1 -> S2     | $k1 * S1$ |]

The model contains two rules:

[{width:25em,=}|  *Type*  |  *Variable*  |  *Formula*          |
|                  Rate | S3         | $k1  *  0.5 10^-3$  |
|                  Rate | S4         | $-k2  *  0.5 10^-3$ |]

The initial conditions are as follows:

[{width:35em,=}|                         |*Value*           |*Units*       |
|Initial amount of S1                |$ 1.5 \x 10^-3$ |mole         |
|Initial amount of S2                |$ 2.0 \x 10^-3$ |mole         |
|Initial amount of S3                |$ 1.5 \x 10^-3$ |mole         |
|Initial amount of S4                |$   4 \x 10^-3$ |mole         |
|Value of parameter k1               |$        0.693$ |second^-1^   |
|Value of parameter k2               |$         0.25$ |second^-1^   |
|Volume of compartment "compartment" |$            10$ |litre        |]

The species have been declared as having substance units only (meaning that
the attribute %hasOnlySubstanceUnits% is set to true). Thus, they must be
treated as amounts where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00333" ];

addCompartment[ compartment, size -> 10 ];
addSpecies[ S1, initialAmount -> 1.5 10^-3, hasOnlySubstanceUnits->True  ];
addSpecies[ S2, initialAmount -> 2.0 10^-3, hasOnlySubstanceUnits->True ];
addSpecies[ S3, initialAmount -> 1.5 10^-3, hasOnlySubstanceUnits->True ];
addSpecies[ S4, initialAmount -> 4 10^-3, hasOnlySubstanceUnits->True ];
addParameter[ k1, value -> .693 ];
addParameter[ k2, value -> .25 ];
addRule[ type->RateRule, variable -> S3, math -> k1  *  0.5 10^-3];
addRule[ type->RateRule, variable -> S4, math -> -k2  *  0.5 10^-3];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 ];

makemodel[]


(* 

category:      Test
synopsis:      Basic two reactions using functiondefinitions with four species in a compartment 
               where the species have only substance units.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Amount, HasOnlySubstanceUnits
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * multiply(S1,S2)$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S3                |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4                |$ 0.5 \x 10^-6$ |mole                      |
|Value of parameter k1               |$  1.3 \x 10^6$ |mole^-1^ second^-1^ |
|Value of parameter k2               |$  0.3 \x 10^6$ |mole^-1^ second^-1^ |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species have been declared as having substance units only. Thus, they 
must be treated as amounts where they appear in expressions.

*)

newcase[ "00115" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6, hasOnlySubstanceUnits->True ];
addSpecies[ S2, initialAmount -> 1.5 10^-6, hasOnlySubstanceUnits->True ];
addSpecies[ S3, initialAmount -> 2.0 10^-6, hasOnlySubstanceUnits->True ];
addSpecies[ S4, initialAmount -> 0.5 10^-6, hasOnlySubstanceUnits->True ];
addParameter[ k1, value -> 1.3 10^6 ];
addParameter[ k2, value -> 0.3 10^6 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * multiply[S1,S2] ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 ];

makemodel[]


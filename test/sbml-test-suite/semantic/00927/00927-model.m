(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, Species, RateRule 
testTags:      Amount, NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying compartment called c and a species
s.  The species is not directly affected by anything but the concentration
will change as a result of the varying compartment size.  However the
amount of the species will remain constant.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | c           | $0.5 * c$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Initial concentration of s    |$2$  |mole          |
|Size of compartment c         |$1$ |litre        |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00927" ];

addCompartment[ c, size -> 1, constant -> False ];
addSpecies[s, initialAmount -> 2];
addRule[ type->RateRule, variable -> c, math -> c * 0.5];

makemodel[]


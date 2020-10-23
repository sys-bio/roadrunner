(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, Parameter, RateRule 
testTags:      NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying 2-dimensional compartment called c and one constant
parameter called k1.  (Note that indeed c is a compartment and not 
species in this model.)

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate                                 | c           | $-c * k1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*    |
|Size of compartment c         |$            0.002$ |metre^2^        |
|Value of parameter k1         |$            1.1$ |second^-1^ |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00906" ];

addCompartment[ c, size -> 0.002, spatialDimensions -> 2, constant -> False];
addParameter[ k1, value -> 1.1 ];
addRule[ type->RateRule, variable -> c, math -> -c * k1];

makemodel[]


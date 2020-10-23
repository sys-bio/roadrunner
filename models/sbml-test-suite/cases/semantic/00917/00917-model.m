(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, RateRule, InitialAssignment 
testTags:      NonConstantCompartment, NonUnityCompartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying 1-dimensional compartment called c.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | c           | $0.5 * c$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | c       | $3 / 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the compartment c has not been
explicitly declared and must be calculated using the InitialAssignment.


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Size of compartment c         |$undeclared$ |metre        |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00917" ];

addCompartment[ c, constant -> False, spatialDimensions -> 1 ];
addInitialAssignment[ c, math -> 2.2/3];
addRule[ type->RateRule, variable -> c, math -> 0.5 * c];

makemodel[]


(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
               0-dimensional compartment using an assignmentRule to vary one species.
componentTags: Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      Amount, 0D-Compartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k2 * S1$  |]

The model contains one rule which assigns value to species S3:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $k1 * S2$  |]

In this case there is no initial value declared for species S3 and thus it
must be calculated by the assignmentRule.  Note that since this
assignmentRule must always remain true, it should be considered during
simulation.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|              Initial amount of S1:| $  1 \x 10^-2$ | mole                      |
|              Initial amount of S2:| $1.5 \x 10^-2$ | mole                      |
|              Initial amount of S3:| $   undeclared$ | mole                      |
|             Value of parameter k1:| $         0.75$ | dimensionless |
|             Value of parameter k2:| $           50$ | second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00156" ];

addCompartment[ compartment, spatialDimensions -> 0];
addSpecies[ S1, initialAmount -> 1 10^-2 ];
addSpecies[ S2, initialAmount -> 1.5 10^-2 ];
addSpecies[ S3];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 50 ];
addRule[ type->AssignmentRule, variable -> S3, math ->k1 * S2];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k2 * S1];

makemodel[]


(* 

category:      Test
synopsis:      Basic single forward reaction involving two species and a
               stoichiometryMath element.
componentTags: StoichiometryMath, Compartment, Species, Reaction, Parameter 
testTags:      Amount, AssignedConstantStoichiometry, DefaultValue, NonUnityStoichiometry
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:   Analytic

Note:  earlier versions of the test suite contained a 3.1 version of this test.
That model was moved to its own test, because it did not have the 'StoichiometryMath'
component.

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and p1.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> (2 * p1)S2 | $k1 * S1 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$          1.5$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter k1               |$          1.5$ |second^-1^ |
|Value of parameter p1               |$            1$ |dimensionless |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00068" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 1.5 ];
addParameter[ p1, value -> 1 ];
addReaction[ reactants->{S1}, products->{S2}, productStoichiometry->{2 * p1},
       reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];

makemodel[]

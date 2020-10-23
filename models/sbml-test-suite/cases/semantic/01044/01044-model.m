(* 

category:      Test
synopsis:      One reversible reaction with three species in one compartment with
an algebraic rule used to determine value of a parameter.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a copy of model 844, with the products and reactants reversed, 
and a negative kinetic law.  This gives the same results, but during the simulation, 
the kinetic law  fails to go negative (a requirement for the ReversibeReaction tag).  
The reaction listed below is therfore equivalent to the one in the file, but reversed from it.

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and two parameters called kf and kr.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 <-> S3 | $(kf * S1 * S2 - kr * S3) * C$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $kf - 0.75$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial amount of S1                |$1.0 \x 10^-2$  |mole                      |
|Initial amount of S2                |$2.0 \x 10^-2$  |mole                      |
|Initial amount of S3                |$1.0 \x 10^-2$  |mole                      |
|Value of parameter kf               |$undeclared$           |litre mole^-1^ second^-1^ |
|Value of parameter kr               |$0.25$           |second^-1^                |
|Volume of compartment C             |$1$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00844" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-2];
addSpecies[ S2, initialAmount -> 2.0 10^-2];
addSpecies[ S3, initialAmount -> 1.0 10^-2];
addParameter[ kf, constant -> False ];
addParameter[ kr, value -> 0.25 ];
addRule[ type->AlgebraicRule, math -> kf - 0.75];
addReaction[ S1 + S2 -> S3,
	     kineticLaw -> (kf * S1 * S2 - kr * S3) * C ];

makemodel[]


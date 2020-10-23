(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| ->S1     | $compartment * k2 * S2$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $S1 + S2-k1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$          0.5$ |mole                      |
|Initial amount of S2                |$          0.5$ |mole                      |
|Value of parameter k1               |$          1.0$ |mole litre^-1^            |
|Value of parameter k2               |$            1$ |second^-1^                |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00039" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.5];
addSpecies[ S2, initialAmount -> 0.5];
addParameter[ k1, value -> 1.0 ];
addParameter[ k2, value -> 1 ];
addRule[ type->AlgebraicRule, math -> S1 + S2-k1];
addReaction[ products->{S1}, modifiers->S2, reversible -> False,
	     kineticLaw -> compartment * k2 * S2 ];

makemodel[]


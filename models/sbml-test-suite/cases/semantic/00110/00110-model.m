(*

category:        Test
synopsis:        Reactions occurring between two compartments with functionDefinitions.
componentTags:   Compartment, FunctionDefinition, Parameter, Reaction, Species
testTags:        Amount, MultiCompartment, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

The model contains two compartments named compartment and compartment1.
There are three species named S1, S2 and S3 and two parameters named k1 and
k2.  Compartment "compartment" contains species S1 and S2.  Compartment
"compartment"1 contains species S3.

The model contains:
* 3 species (S1, S2, S3)
* 2 parameters (k1, k2)
* 2 compartments (compartment, compartment1)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| reaction1: S1 + S2 -> 2S2 | $compartment * S2 * multiply(k1, S1)$ |
| reaction2: S2 -> S3 | $compartment1 * k2 * (S2 + -1 * S3)$ |]

The first reaction occurs entirely in compartment "compartment", whereas
the second reaction occurs between a species in "compartment" and a species
in "compartment1".

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial amount of species S2 | $1$ | variable |
| Initial amount of species S3 | $0$ | variable |
| Initial value of parameter k1 | $0.75$ | constant |
| Initial value of parameter k2 | $0.25$ | constant |
| Initial volume of compartment 'compartment' | $1$ | constant |
| Initial volume of compartment 'compartment1' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

newcase[ "00110" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, size -> 1 ];
addCompartment[ compartment1, size -> 1 ];
addSpecies[ S1, compartment->compartment, initialAmount -> 1.0];
addSpecies[ S2, compartment->compartment, initialAmount -> 1.0];
addSpecies[ S3, compartment->compartment1, initialAmount -> 0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addReaction[ S1 + S2 -> S2  +  S2, reversible -> False,
	     kineticLaw -> multiply[k1,S1] * S2 * compartment ];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k2 * (S2-S3) * compartment1 ];

makemodel[]


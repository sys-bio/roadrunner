(*

category:        Test
synopsis:        A 'kitchen sink' model with 34 tags.  Woo-hoo!
componentTags:   AssignmentRule, CSymbolAvogadro, CSymbolTime, Compartment, EventNoDelay, EventPriority, EventWithDelay, FunctionDefinition, InitialAssignment, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, AssignedVariableStoichiometry, BoundaryCondition, ConstantSpecies, ConversionFactors, EventIsNotPersistent, EventIsPersistent, EventT0Firing, EventUsesAssignmentTimeValues, EventUsesTriggerTimeValues, HasOnlySubstanceUnits, InitialValueReassigned, LocalParameters, MultiCompartment, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, NonUnityStoichiometry, ReversibleReaction, SpeciesReferenceInMath
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 This is a 'kitchen sink' model with almost everything in it.  It's a RoadRunner-friendly model in that it only doesn't have bits that RoadRunner can't handle:  AlgebraicRules, CSymbolDelays, and FastReactions.  It also does not test RandomEventExecution because those tests are fundamentally difficult to put together, and inherently non-precise in their output.  It is, frankly, a fairly ridiculous model with things thrown in willy-nilly with no regard to unit correctness in the interest of simply testing everything together.  The canonical results were taken from RoadRunner output.

The model contains:
* 4 species (S1, S2, S3, S4)
* 8 parameters (kavo, k1, k3, k4, k5, k2, conversion1, conversion2)
* 2 compartments (comp, comp2)

It also contains 1 function definition(s):
; kinetics: $(-a + b + c + d + e + f + g + h + i + j) / 10$

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| _J0: S3 + S1ref S1 -> S2ref S2 | $kinetics(k1, k2, k3, k4, k5, S1, S1ref, S3, S4, S2)$ |]
Note:  the following stoichiometries are set separately:  S1ref, S2ref


There are 8 events:

[{width:55em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  |  *initialValue*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| _E0 | $geq(time, 0.5)$ | (unset) | true | true | Trigger time | $0$ | $k4 = 0.1$ |
| _E1 | $geq(time, 1)$ | $1$ | true | true | Trigger time | $2$ | $k3 = 4$ |
| _E2 | $geq(time, 1)$ | $2$ | true | true | Assignment time | $2$ | $k4 = k3$ |
| _E3 | $geq(time, 1)$ | $3$ | true | true | Trigger time | $2$ | $k5 = k4$ |
| _E4 | $geq(time, 1)$ | $4$ | true | true | Assignment time | $2$ | $k3 = k5$ |
| _E5 | $leq(comp, 5.1)$ | (unset) | true | false | Trigger time | $1$ | $S3 = 4$ |
| _E6 | $and(geq(time, 1), leq(time, 4))$ | (unset) | true | true | Trigger time | $4.995$ | $k4 = 14.5$ |
| _E7 | $and(geq(time, 1), leq(time, 4))$ | (unset) | false | true | Trigger time | $4.995$ | $k5 = 0$ |]


There are 4 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | comp | $1$ |
| Assignment | comp2 | $k4$ |
| Rate | k4 | $time$ |
| Assignment | S2ref | $k1 * S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S4 | $2$ | constant |
| Initial amount of species S1 | $1$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial amount of species S3 | $4$ | variable |
| Initial value of parameter kavo | $avogadro / 6.022000e+023$ | constant |
| Initial value of parameter k1 | $1.1$ | constant |
| Initial value of parameter k2 | $8.12$ | constant |
| Initial value of parameter conversion1 | $10$ | constant |
| Initial value of parameter conversion2 | $100$ | constant |
| Initial value of parameter k3 | $2.5$ | variable |
| Initial value of parameter k4 | $1$ | variable |
| Initial value of parameter k5 | $2.8$ | variable |
| Initial volume of compartment 'comp' | $5$ | variable |
| Initial volume of compartment 'comp2' | $k4$ | variable |]

*)


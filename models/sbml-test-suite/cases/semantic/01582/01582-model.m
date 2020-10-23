(*

category:        Test
synopsis:        A non-persistent event whose delayed assignment would affect a reaction's stoichiometry.
componentTags:   AssignmentRule, Compartment, EventWithDelay, Parameter, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, EventIsNotPersistent, InitialValueReassigned, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The event in this model is not persistent, and therefore does not fire because its trigger reverts to 'false' before executing.  It would assign a value to a parameter, which in turn is assigned to the stoichiometry of the reaction; this does not happen.

The model contains:
* 1 species (S1)
* 2 parameters (k1, P1)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_stoich S1 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S1_stoich


There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Delay*  | *Event Assignments* |
| E0 | $(k1 > 4.5) && (k1 < 5.1)$ | false | $2$ | $P1 = 3$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k1 | $1$ |
| Assignment | S1_stoich | $P1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of parameter P1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $P1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

(*

category:        Test
synopsis:        A hierarchical model with time, extent, and replacement conversion factors.
componentTags:   AssignmentRule, Compartment, EventNoDelay, EventPriority, Parameter, RateRule, Reaction, Species, comp:Deletion, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        Amount, InitialValueReassigned, NonConstantParameter, comp:ConversionFactor, comp:ExtentConversionFactor, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 Time, extent, and replacement conversion factors all combine in this model to affect a variety of constructs.

The 'flattened' version of this hierarchical model contains:
* 1 species (A__S1)
* 10 parameters (extentconv, timeconv, Q, R, cf, X, U1, U2, A__k1, A__y)
* 1 compartment (A__default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A___J0: A__S1 -> | $extentconv / timeconv * A__k1 * A__S1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| A___E0 | $gt(X / cf, 0.1515)$ | $Q$ | $A__y = 4$ |
| A___E1 | $gt(X / cf, 0.1515)$ | $R$ | $A__y = 5$ |]


There are 4 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | U1 | $X / 100$ |
| Assignment | U2 | $23$ |
| Assignment | Q | $X / cf - 3$ |
| Rate | X | $0.2 * cf / timeconv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A__S1 | $7.2$ | variable |
| Initial value of parameter extentconv | $10$ | constant |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter R | $20$ | constant |
| Initial value of parameter cf | $100$ | constant |
| Initial value of parameter A__k1 | $1.06$ | constant |
| Initial value of parameter Q | $X / cf - 3$ | variable |
| Initial value of parameter X | $15$ | variable |
| Initial value of parameter U1 | $1.1$ | variable |
| Initial value of parameter U2 | $23$ | variable |
| Initial value of parameter A__y | $8.8$ | variable |
| Initial volume of compartment 'A__default_compartment' | $1$ | constant |]

*)


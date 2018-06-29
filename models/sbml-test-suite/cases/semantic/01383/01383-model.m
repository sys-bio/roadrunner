(*

category:        Test
synopsis:        A replaced species reference whose ID is used in am event priority.
componentTags:   CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, NonConstantParameter, NonUnityStoichiometry, SpeciesReferenceInMath, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced species reference (itself in a replaced reaction) is used in an event priority.

The 'flattened' version of this hierarchical model contains:
* 1 species (S1)
* 1 parameter (A__x)
* 1 compartment (C)
* 1 species reference (J1_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: -> 5S1 | $1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| A__E0 | $time > 5.5$ | $J1_sr$ | $A__x = 4$ |
| A__E1 | $time > 5.5$ | $4$ | $A__x = 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial value of parameter A__x | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

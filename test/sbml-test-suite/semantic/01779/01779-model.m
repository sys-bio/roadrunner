(*

category:        Test
synopsis:        A species concentration and its compartment assigned to by the same event.
componentTags:   AssignmentRule, CSymbolTime, Compartment, EventNoDelay, Parameter, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 A species in a non-unity compartment is set 'hasOnlySubstanceUnits=false', meaning that the species concentration is used in all situations, including the event assignment.  However, another event assignment in the same event is to the volume of that species' compartment.  Both assignments are made based on the current state of the model, and *not* of of the state of the model after any one of the assignments.  As a result, the assignment to S1 should set the amount of S1 to 0.1.  The simultaneous assignment of 0.2 to the compartment means that the concentration of S1 ends up being 0.5 (0.1/0.2).  This is counter-intuitive, since the concentration of S1 was just assigned to be 0.2!  But it is a consequence of S1's concentration being derived instead of atomic.

This test is a simpler variant of 01506, when it was realized some of the subtleties of that test were complicated by the rate rules.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time >= 0.45$ | $S1 = 0.2$ |
|  |  | $C1 = 0.2$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $2$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

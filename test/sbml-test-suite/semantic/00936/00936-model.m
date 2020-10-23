(*

category:      Test
synopsis:      Limited-time oscillation that triggers a delayed event three times before the first one executes.
componentTags: AssignmentRule, CSymbolTime, Compartment, EventWithDelay, Species
testTags:      Concentration, BoundaryCondition, EventIsPersistent, EventUsesAssignmentTimeValues, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 The model contains one species (S1) that oscillates (from the 'sin' function) for two seconds, then stops.  During that oscillation, the single event is triggered three times with a delay of two seconds each, increasing the value of the second species (S2) by one each time.  As 'useValuesFromTriggerTime' is false, this means that first S1 oscillates, then S2 steps up three times.

The model contains:
* 2 species (S1, S2)
* 1 compartment (default_compartment)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| _E0 | $S1 < 0$ | Assignment time | $2$ | $S2 = S2 + 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S1 | $piecewise(sin(time * 10), time < 2, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $piecewise(sin(time * 10), time < 2, 1)$ | variable |
| Initial concentration of species S2 | $0$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


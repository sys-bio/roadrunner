(*

category:        Test
synopsis:        An event assignment using the assignment-time value of a substance-only species.
componentTags:   CSymbolTime, Compartment, EventWithDelay, RateRule, Species
testTags:        Amount, EventUsesAssignmentTimeValues, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event assignment uses the ID of a substance-only speces, using the value from the assignment time.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E1 | $time >= 2.5$ | Assignment time | $2$ | $S1 = S1 + 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $-1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $10$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

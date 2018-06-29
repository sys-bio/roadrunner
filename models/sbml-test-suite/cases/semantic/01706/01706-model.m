(*

category:        Test
synopsis:        An event assignment using the trigger-time value of a substance-only species.
componentTags:   CSymbolTime, Compartment, EventWithDelay, RateRule, Species
testTags:        Amount, EventUsesTriggerTimeValues, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event assignment uses the ID of a substance-only speces, using the value from when the trigger time.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E1 | $time >= 2.5$ | $2$ | $S1 = S1 + 1$ |]


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

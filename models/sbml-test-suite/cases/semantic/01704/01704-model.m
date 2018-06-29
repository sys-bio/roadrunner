(*

category:        Test
synopsis:        A non-persistent event involving a substance-only species.
componentTags:   Compartment, EventWithDelay, RateRule, Species
testTags:        Amount, EventIsNotPersistent, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a delayed event is triggered by a substance-only species entering a window of values which will be different if the species is mis-interpreted in that section as a concentration.  The event is not persistent, so when the trigger reverts to 'false' before being executed, the execution is cancelled.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one event:

[{width:45em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E0 | $(S1 <= 4.3) && (S1 >= 3)$ | false | Assignment time | $2$ | $S1 = 10$ |]


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

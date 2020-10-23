(*

category:      Test
synopsis:      Test a delayed event trigger that transitions from false to true at t0.
componentTags: Compartment, EventWithDelay, RateRule, Species
testTags:      Amount, EventT0Firing, HasOnlySubstanceUnits, NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

 This model has a delayed event which fires at t0 due to it being set 'initialValue=false'.

The model contains:
* 1 species (S3)
* 1 compartment (comp)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  |  *Delay*  | *Event Assignments* |
| E1 | $comp &leq; 5.1$ | false | $1.05$ | $S3 = 4$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | comp | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S3 | $2$ | variable |
| Initial volume of compartment 'comp' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


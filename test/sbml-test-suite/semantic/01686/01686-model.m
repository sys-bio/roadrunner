(*

category:        Test
synopsis:        Model conversion factor for events that fire at t0.
componentTags:   Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, EventT0Firing
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a model conversion factor is set for species reset at t0 by an event.

The model contains:
* 2 species (S1, S2)
* 1 parameter (m_cf)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.01$ |]

The conversion factor for the model is 'm_cf'


There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $S1 > 0.9$ | false | $S2 = 3$ |
|  |  |  | $S1 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial amount of species S2 | $0$ | variable |
| Initial value of parameter m_cf | $3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

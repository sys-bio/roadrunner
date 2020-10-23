(*

category:      Test
synopsis:      A species that changes behavior between its IntialAssignment andsubsequently produced by a reaction, being echoed by a parameter viewing it through a delay.
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:      Amount, NonConstantParameter, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains a single species S1 whose InitialAssignment claims that it has been oscillating (the sine function) up until the start of the simulation, at which point a reaction begins to create it.  A parameter (y) echoes this species with a 0.2 second delay, meaning that it starts by 'seeing' the oscillations before echoing the linear increase.

The model contains:
* 1 species (x)
* 1 parameter (y)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> x | $1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | y | $2 + delay(x, 0.2)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species x | $1 * sin(10 * time)$ | variable |
| Initial value of parameter y | $2 + delay(x, 0.2)$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


(*

category:      Test
synopsis:      A variable species that varies in its IntialAssignment, but is constant thereafte, being echoed by a parameter viewing it through a delay.
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, Compartment, InitialAssignment, Parameter, Species
testTags:      Amount, NonConstantParameter, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains a single species x whose InitialAssignment claims that it has been oscillating (the sine function) up until the start of the simulation, at which point it stops changing.  A parameter (y) echoes this species with a 0.5 second delay, meaning that it starts by 'seeing' the oscillations before flattening out.

The model contains:
* 1 species (x)
* 1 parameter (y)
* 1 compartment (default_compartment)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | y | $2 + delay(x, 0.5)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species x | $1 * sin(10 * time)$ | variable |
| Initial value of parameter y | $2 + delay(x, 0.5)$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor affecting many components.
componentTags:   AlgebraicRule, AssignmentRule, CSymbolDelay, CSymbolTime, Compartment, EventWithDelay, InitialAssignment, Parameter, RateRule, Reaction, Species, comp:ModelDefinition, comp:ReplacedBy, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 This model contains a submodel with various time-based constructs in it, contained in a model with a time conversion factor.  For broken-down versions of this model that test only one construct at a time, see tests 01170-01177.

The 'flattened' version of this hierarchical model contains:
* 1 species (s1)
* 6 parameters (timeconv, t1, t2, t3, t4, t5)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| sub1__J0: -> s1 | $1 / timeconv * (t3 * (time / timeconv) / (s1 * delay(t5, timeconv * 2.000000e-001)))$ |]


There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| event_0 | $gt(time / timeconv, 3)$ | $timeconv * (1 / (time / timeconv))$ | $t5 = time / timeconv$ |]


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | t1 | $(time / timeconv / t1 + 3) / timeconv$ |
| Assignment | t3 | $delay(t1, timeconv * 3)$ |
| Algebraic | $0$ | $t4 - delay(t3, timeconv * (time / timeconv / 2))$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species s1 | $0.001$ | variable |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t1 | $1$ | variable |
| Initial value of parameter t2 | $time / timeconv + 3$ | variable |
| Initial value of parameter t3 | $delay(t1, timeconv * 3)$ | variable |
| Initial value of parameter t4 | $1$ | variable |
| Initial value of parameter t5 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)


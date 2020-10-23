(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor.
componentTags:   CSymbolTime, EventWithDelay, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        EventUsesTriggerTimeValues, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

This model contains a submodel with an event with various time-based constructs in it, contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (timeconv, t5)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
|  | $gt(time / timeconv, 3)$ | $timeconv * (1 / (time / timeconv))$ | $t5 = time / timeconv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t5 | $1$ | variable |]

*)


(*

category:        Test
synopsis:        A hierarchical model with a conversion factor affecting an event assignment.
componentTags:   CSymbolTime, EventNoDelay, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        NonConstantParameter, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The replaced parameter in the submodel is assigned a value by an event.  The replacement contains a conversion factor, affecting the value given by the event.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (conv, p8)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| sub1__E1 | $gt(time, 0.31)$ | $p8 = (800 + p8 / conv) * conv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter conv | $0.1$ | constant |
| Initial value of parameter p8 | $8$ | variable |]

*)


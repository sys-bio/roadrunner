(* 

category:      Test
synopsis:      Single forward reaction with two species in one compartment and an event
componentTags: Compartment, CSymbolTime, Species, EventNoDelay, EventPriority
testTags:      Amount
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "C".  There are two
species called S1 and S2. 


The model contains two events that assign value to species S1, S2 and S3 defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Priority* | *Delay* | *Assignments* |
 | Event1 | $t >= 0.99$ | $1$  | $-$   | $S1 = 4$    |
 |        |          |      |       | $S2 = 5$    |
 |        |          |      |       | $S3 = 6$    |
 | Event2 | $t >= 0.99$ | $0$  | $-$   | $S1 = 1$    |
 |        |          |      |       | $S2 = 2$    |
 |        |          |      |       | $S3 = 3$    |]
 
The Events will fire simultaneously and must be executed in order determined by
the Priority. Thus the assignments of Event1 will be executed followed by the 
assignments for Event2.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$0$  |mole           |
|Initial amount of S2                |$0$              |mole           |
|Initial amount of S2                |$0$              |mole           |
|Volume of compartment "C" |$1$              |litre          |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but (as
per usual SBML principles) their symbols represent their values in
concentration units where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)



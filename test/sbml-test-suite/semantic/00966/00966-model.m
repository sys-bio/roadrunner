(*

category:      Test
synopsis:      Competing events with the same priority, jointly causing a parameter to monotonically increase, calculating a chi-squared statistic to ensure randomness.  NOTE:  STOCHASTIC TEST. This test is designed to fail with a frequency of 0.00002, but it still may happen.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, EventPriority, FunctionDefinition, Parameter
testTags:      EventIsNotPersistent, EventIsPersistent, EventUsesTriggerTimeValues, NonConstantParameter, RandomEventExecution
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model (the same basic model as 952) contains two events with the same trigger, the same priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.

In this test, we also keep track of each run of 16 tests in a row, calculating the sum of Q during that run, and incrementing the appropriate frequency (chi0 -> chi16).  At the end of 6,420 tests, the chi-square value is calculated from the observed and the pre-calculated expected frequencies of the 17 possible values.  If this value is greater than 50, an error is triggered.  The liklihood of the error being triggered is 0.00002, so it is quite unlikely (though possible) that a simulator would trigger it with a working random number generator, and almost impossible for this to happen twice in a row.

This model has a number of events, so it's probably best to look at the antimony file for more details:  00966-antimony.txt


The model contains:
* 27 parameters (S, Q, R, reset, chicount, chitest, assemble, chi0, chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, chi9, chi10, chi11, chi12, chi13, chi14, chi15, chi16, tests, finalchi, error)

It also contains 1 function definition(s):
; calculateerror: $pow(x0 - 0.0952, 2) / 0.0952 + pow(x1 - 1.523, 2) / 1.523 + pow(x2 - 11.428, 2) / 11.428 + pow(x3 - 53.32, 2) / 53.32 + pow(x4 - 173.29, 2) / 173.29 + pow(x5 - 415.898, 2) / 415.898 + pow(x6 - 762.48, 2) / 762.48 + pow(x7 - 1089.26, 2) / 1089.26 + pow(x8 - 1225.415, 2) / 1225.415 + pow(x9 - 1089.26, 2) / 1089.26 + pow(x10 - 762.48, 2) / 762.48 + pow(x11 - 415.898, 2) / 415.898 + pow(x12 - 173.29, 2) / 173.29 + pow(x13 - 53.32, 2) / 53.32 + pow(x14 - 11.428, 2) / 11.428 + pow(x15 - 1.523, 2) / 1.523 + pow(x16 - 0.0952, 2) / 0.0952$

There are 23 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Rinc | $geq(time - reset, 0.01)$ | $10$ | false | $reset = time$ |
|  |  |  |  | $chicount = chicount + 1$ |
|  |  |  |  | $R = R + 0.01$ |
| Qinc | $geq(time - reset, 0.01)$ | $10$ | false | $reset = time$ |
|  |  |  |  | $chitest = chitest + 1$ |
|  |  |  |  | $chicount = chicount + 1$ |
|  |  |  |  | $Q = Q + 0.01$ |
| repeat | $geq(floor(chicount / 16), 1)$ | $5$ | true | $assemble = 1.5$ |
|  |  |  |  | $chicount = 0$ |
| _E0 | $and(geq(assemble, 1), eq(chitest, 0))$ | $3$ | true | $chi0 = chi0 + 1$ |
| _E1 | $and(geq(assemble, 1), eq(chitest, 1))$ | $3$ | true | $chi1 = chi1 + 1$ |
| _E2 | $and(geq(assemble, 1), eq(chitest, 2))$ | $3$ | true | $chi2 = chi2 + 1$ |
| _E3 | $and(geq(assemble, 1), eq(chitest, 3))$ | $3$ | true | $chi3 = chi3 + 1$ |
| _E4 | $and(geq(assemble, 1), eq(chitest, 4))$ | $3$ | true | $chi4 = chi4 + 1$ |
| _E5 | $and(geq(assemble, 1), eq(chitest, 5))$ | $3$ | true | $chi5 = chi5 + 1$ |
| _E6 | $and(geq(assemble, 1), eq(chitest, 6))$ | $3$ | true | $chi6 = chi6 + 1$ |
| _E7 | $and(geq(assemble, 1), eq(chitest, 7))$ | $3$ | true | $chi7 = chi7 + 1$ |
| _E8 | $and(geq(assemble, 1), eq(chitest, 8))$ | $3$ | true | $chi8 = chi8 + 1$ |
| _E9 | $and(geq(assemble, 1), eq(chitest, 9))$ | $3$ | true | $chi9 = chi9 + 1$ |
| _E10 | $and(geq(assemble, 1), eq(chitest, 10))$ | $3$ | true | $chi10 = chi10 + 1$ |
| _E11 | $and(geq(assemble, 1), eq(chitest, 11))$ | $3$ | true | $chi11 = chi11 + 1$ |
| _E12 | $and(geq(assemble, 1), eq(chitest, 12))$ | $3$ | true | $chi12 = chi12 + 1$ |
| _E13 | $and(geq(assemble, 1), eq(chitest, 13))$ | $3$ | true | $chi13 = chi13 + 1$ |
| _E14 | $and(geq(assemble, 1), eq(chitest, 14))$ | $3$ | true | $chi14 = chi14 + 1$ |
| _E15 | $and(geq(assemble, 1), eq(chitest, 15))$ | $3$ | true | $chi15 = chi15 + 1$ |
| _E16 | $and(geq(assemble, 1), eq(chitest, 16))$ | $3$ | true | $chi16 = chi16 + 1$ |
| _E17 | $geq(assemble, 1)$ | $1$ | true | $tests = tests + 1$ |
|  |  |  |  | $chitest = 0$ |
|  |  |  |  | $assemble = 0$ |
| _E18 | $and(geq(tests, 6240), eq(finalchi, 0))$ | $0$ | true | $finalchi = calculateerror(chi0, chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, chi9, chi10, chi11, chi12, chi13, chi14, chi15, chi16)$ |
| error_check | $gt(finalchi, 50)$ | $-1$ | true | $error = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S | $Q + R$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S | $Q + R$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter chicount | $0$ | variable |
| Initial value of parameter chitest | $0$ | variable |
| Initial value of parameter assemble | $0$ | variable |
| Initial value of parameter chi0 | $0$ | variable |
| Initial value of parameter chi1 | $0$ | variable |
| Initial value of parameter chi2 | $0$ | variable |
| Initial value of parameter chi3 | $0$ | variable |
| Initial value of parameter chi4 | $0$ | variable |
| Initial value of parameter chi5 | $0$ | variable |
| Initial value of parameter chi6 | $0$ | variable |
| Initial value of parameter chi7 | $0$ | variable |
| Initial value of parameter chi8 | $0$ | variable |
| Initial value of parameter chi9 | $0$ | variable |
| Initial value of parameter chi10 | $0$ | variable |
| Initial value of parameter chi11 | $0$ | variable |
| Initial value of parameter chi12 | $0$ | variable |
| Initial value of parameter chi13 | $0$ | variable |
| Initial value of parameter chi14 | $0$ | variable |
| Initial value of parameter chi15 | $0$ | variable |
| Initial value of parameter chi16 | $0$ | variable |
| Initial value of parameter tests | $0$ | variable |
| Initial value of parameter finalchi | $0$ | variable |
| Initial value of parameter error | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)


The SBML Test Suite – Semantic Test Cases
==========================================

The semantic test cases portion of the SBML Test Suite contains valid SBML models together with expected numerical results when these models are simulated using a deterministic simulation approach.  (Stochastic simulations are tested using a separate, companion set of test cases in the SBML Test Suite.)  An example of a deterministic simulator is a system using a numerical differential-algebraic solver that supports discontinuous events.  Each test consists of a directory containing the model (or models, if that model can be translated to other SBML Levels + Version combinations without semantic loss), together with instructions on how to simulate that model, and the expected results.

----
*Main Authors*: Sarah M. Keating<sup>a,b</sup>, Lucian P. Smith<sup>b,c</sup>, Bruce Shapiro <sup>b</sup>, Michael Hucka<sup>b</sup>, Frank T. Bergmann<sup>d</sup>, Brett Olivier<sup>e</sup>, Andrew Finney<sup>b</sup>

*Institutions*:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>a</sup> EMBL-EBI, Hinxton, Cambridgeshire, UK<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>b</sup> California Institute of Technology, Pasadena, CA, US<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>c</sup> University of Washington, Seattle, WA, US<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>d</sup> University of Heidelberg, Heidelberg, DE<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>e</sup> Vrije Universiteit, Amsterdam<br>

*Repository*:   [https://github.com/sbmlteam/sbml-test-suite](https://github.com/sbmlteam/sbml-test-suite)

*Bug tracker*:   [https://github.com/sbmlteam/sbml-test-suite/issues](https://github.com/sbmlteam/sbml-test-suite/issues)

*Developers' discussion group*: [https://groups.google.com/forum/#!forum/moccasin-dev](https://groups.google.com/forum/#!forum/sbml-interoperability)

*Pivotal tracker*: [https://www.pivotaltracker.com/n/projects/68714](https://www.pivotaltracker.com/n/projects/68714)


Organization of folders and files
---------------------------------

Relative to the entire SBML Test Suite, the directory where this `README.md` file is located is `cases/semantic`.  In this directory, you will find a large number of subdirectories whose names are all five digits, i.e., _NNNNN_.  Each contains the files for a single test case.  The following are the files provided:

* `NNNNN-sbml-lXvY.xml`.  These are the SBML files defining the model that constitutes a given test.  Most tests have all of the following files, but note that they all define the same test; they are simply in different SBML Level/Version formats:

  * `NNNNN-sbml-l3v2.xml`   – SBML Level 3 Version 2
  * `NNNNN-sbml-l3v1.xml`   – SBML Level 3 Version 1
  * `NNNNN-sbml-l2v5.xml`   – SBML Level 2 Version 5
  * `NNNNN-sbml-l2v4.xml`   – SBML Level 2 Version 4
  * `NNNNN-sbml-l2v3.xml`   – SBML Level 2 Version 3
  * `NNNNN-sbml-l2v2.xml`   – SBML Level 2 Version 2
  * `NNNNN-sbml-l2v1.xml`   – SBML Level 2 Version 1
  * `NNNNN-sbml-l1v2.xml`   – SBML Level 1 Version 2

* `NNNNN-model.m`.  A description of the test model that includes the "tags" used to describe the test being performed, as well as a description of the model.  The format of this file is described in a separate section below.  This file is also used to generate the human-readable HTML file, and in some models, it contains commands used to generate the SBML file as well, but this is not available for all test case files.  Note that the simulation settings are stored in a separate file, named `NNNNN-settings.txt`.

* `NNNNN-settings.txt`.  This is a single file listing the general simulation control values that should be used when running the given model.  The format of this file is described in a separate section below.

* `NNNNN-model.html`.  A brief, nicely-formatted, human-readable description of the purpose of this specific test.

* `NNNNN-results.csv`.  The results expected from simulating the model. The file format is simple comma-separated values.  Approximately 1/5 of cases have results generated from an analytical solution to the model; the rest come from numerical solutions produced agreed-upon by at least two different SBML-compatible tools.

* `NNNNN-results.xlsx`.  (Some models only.) The results of using an analytical function in Microsoft Excel to produce the results for the model. This is in turn used to produce the `NNNNN-results.csv` file for those models.

* `NNNNN-plot.jpg`.  A plot of the expected time-course simulation results in JPEG format.

* `NNNNN-sbml-lXvY-sedml.xml`.  These are files in SED-ML format for running the test case in software systems that can automate their execution using SED-ML.  Like the SBML files themselves, these come in different SBML Level + Version combinations.

  * `NNNNN-sbml-l3v2-sedml.xml`   – SBML Level 3 Version 2
  * `NNNNN-sbml-l3v1-sedml.xml`   – SBML Level 3 Version 1
  * `NNNNN-sbml-l2v5-sedml.xml`   – SBML Level 2 Version 5
  * `NNNNN-sbml-l2v4-sedml.xml`   – SBML Level 2 Version 4
  * `NNNNN-sbml-l2v3-sedml.xml`   – SBML Level 2 Version 3
  * `NNNNN-sbml-l2v2-sedml.xml`   – SBML Level 2 Version 2
  * `NNNNN-sbml-l2v1-sedml.xml`   – SBML Level 2 Version 1
  * `NNNNN-sbml-l1v2-sedml.xml`   – SBML Level 1 Version 2

* `NNNNN-antimony.txt`.  (Some models only.)  A description of the test model in Antimony format used to generate the SBML file.

* `NNNNN.omex`.  A [COMBINE Archive](http://co.mbine.org/standards/omex) of the test case, structured according to the _Open Modeling EXchange format_ (OMEX).  This archive contains the SBML model files, the SED-ML files, the JPG plot file, the results CSV file, and the model description file in HTML format, together with some metadata and a manifest.  These files are used by software that can read COMBINE Archives.


The format of the model description (`.m`) file
-----------------------------------------------

The files whose name have the form `NNNNN-model.m` contain a description of the model.  The description portion is at the beginning of the file, enclosed in Mathematica comment delimiters (the `(*` and `*)` character sequences), and after the `*)`, there may be commands used to generate the actual SBML file.  The most important part of this file is at the beginning.  Here is an example:

```
category:        Test
synopsis:        Basic single forward reaction with two species.
componentTags:   Compartment, Species, Reaction, Parameter 
testTags:        InitialAmount
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent:
```

The meaning of each field is described below:

* `category`: Many of the models used in the test suite are not biologically meaningful.  This field determines whether this case refers to a test model or a more realistic model.  `Test` means the model is for testing and not meant to be biologically realistic.

* `synopsis`: Brief textual description of the model, in English.  The value of this field may span over more than one line.

* `componentTags`: These tags describe the SBML components that are present in this model.  Tags are discussed in a separate section below.
  
* `testTags`: These tags describe the aspects of SBML interpretation that are being tested in this model.  Tags are discussed in a separate section below.

* `testType`: Since it is possible to simulate data from models in different ways, this tag indicates the type of test to perform on the given model.  Currently, there are two tags: `TimeCourse`, for the majority of tests in the suite, and `FluxBalanceSteadyState`, for tests that involve the SBML Level 3 'Flux Balance Analysis' package.
  
* `levels`: Not all SBML components and attributes exist in every level and version of SBML.  This tag indicates the relevant SBML Levels+Version combinations permissible for this particular case.  The format of the tag is two integers separated by a dot; e.g., `2.4` signifies Level 2 Version 4.

* `generatedBy`: This tag indicates whether the results data for this case has been generated analytically (tag value `Analytic`) or numerically (tag value `Numeric`).

* `packagesPresent`: This tag indicates whether any SBML Level 3 packages are present in the model.  Possible tag values are (at present) `comp`, if elements from the Hierarchical Model Composition package are present, `fbc`, if elements from the Flux Balance Constraints package are present, `fbc_v1` if the Flux Balance Constraints package being used is version 1, and `fbc_v2` if version 2.  All Flux Balance Constraints packages will therefore have at least two tags: one `fbc` tag, and one tag indicating which version of that specification is being used.


The format of the settings file
-------------------------------

The information about run parameters is stored in a file named `NNNNN-settings.txt`.  This file is generated automatically after a test case is created by an author.  The generation is accomplished using a script that reads the CSV file and summarizes the actual test run; this approach reduces the chances of human errors introducing a mismatch between the claimed simulation settings and the actual settings used to generate the reference data.

The format of the file is very simple.  Here's an example:

```
start: 0
duration: 5.0
steps: 50
variables: S1, S2
absolute: 1e-12
relative: 0.0001
amount: S1, S2
concentration:
```

Depending on the type of test requested (`TimeCourse` or `FluxBalanceSteadyState`) this data will have slightly different meanings


### Settings files for TimeCourse tests

The information about run parameters is stored in a file named `NNNNN-settings.txt`.  This file is generated automatically after a test case is created by an author.  The generation is accomplished using a script that reads the CSV file and summarizes the actual test run; this approach reduces the chances of human errors introducing a mismatch between the claimed simulation settings and the actual settings used to generate the reference data.

* `start`: The start time of the simulation time-series data in the output (CSV) file.  Often this is `0`, but not necessarily.

* `duration`: The duration of the simulation run, in the model's units of time.

* `steps`: The number of steps at which the output is sampled.  The samples are evenly spaced.  When a simulation system calculates the data points to record, it will typically divide the duration by the number of time steps.  Thus, for _X_ steps, the data file will have _X_+1 data rows.

* `variables`: The variables (in addition to time) whose values are tabulated in the CSV file.  These are SBML model id's separated by commas.  The order of this list is significant: a results file without headers will be assumed to contain values for variables in the order present on this line.  Important note #1: if a symbol in this list refers to a species in the model, then that symbol will ALSO be listed in either the `amount` or `concentration` lists described below.  The presence of a species in the `amount` or `concentration` list indicates whether the values for the species are considered to be in amount units (i.e., direct quantity, not a concentration) or in concentration units, respectively.  Important note #2: if a listed variable has two underscores (`__`) in its name, that variable is actually present only in a submodel of the main model, using the SBML Level 3 Hierarchical Model Composition package, in the format `submodelID__variableID`.

* `absolute`: A float-point number representing the absolute difference permitted for this test case when comparing numerical values to the results to those produced by a software tool.  The meaning of this tolerance and the formula used to calculate data point differences is discussed below.

* `relative`: A float-point number representing the relative difference permitted for this test case when comparing numerical values to the results produced by a software tool.  The value of 0.0001 was the tolerance agreed upon by the SBML community during discussions at SBML Hackathons in 2008.  The meaning of this tolerance and the formula used to calculate data point differences is discussed below.

* `amount`: A list of the variable whose output in the results file is in amount (not concentration) units.  This list of variables must be a subset of the names listed in `variables`.

* `concentration`: A list of the variable whose output in the results file is in concentration (not amount) units.  This list of variables must be a subset of the names listed in `variables`.


### Tolerances and errors for TimeCourse tests

Due to the nature of digital computing, the numerical results produced by any software can never be expected to be exactly the same as the reference results provided with each test case.  Comparison of results must be made with respect to some bounds or tolerances.  For purposes of comparing the results produced by software to the reference results provided in this test suite, two tolerances are defined: absolute and relative.  These are defined in the settings file for each test case (see above).

Let the following variables be defined:

* <i>T<sub>a</sub></i> stand for the absolute tolerance for a test case,

* <i>T<sub>r</sub></i> stand for the relative tolerance for a test case,

* <i>C<sub>ij</sub></i> stand for the expected correct value for row <i>i</i>, column <i>j</i>, of the result data set for the test case
  
* <i>U<sub>ij</sub></i> stand for the corresponding value produced by a given software simulation system run by the user

These absolute and relative tolerances are used in the following way: a data point <i>U<sub>ij</sub></i> is considered to be within tolerances if and only if the following expression is true:
  
<p align="center">
<i>|C<sub>ij</sub> - U<sub>ij</sub>| &le; (T<sub>a</sub> + T<sub>r</sub> * |C<sub>ij</sub>|)</i>
</p>


### Settings file for FluxBalanceSteadyState tests 

For tests with the `testType` tag `FluxBalanceSteadyState`, the settings file is the same format as for `TimeCourse` tests, but more lines are left blank, as they have no meaning.  Here's an example:

```
start:
duration:
steps:
variables: J0, J1, OBJF
absolute: 0.001
relative: 0.001
amount:
concentration:
```

The `variables` line still indicates which variable values are to be output after steady state is reached, and the `relative` and `absolute` lines indicate the relative and absolute error allowed for each data point.

The other lines are irrelevant to a flux balance steady state simulation: because the system is being analyzed at steady state, there is no `start`, `duration`, or `steps` to be taken.  Because only fluxes and objective functions are being analyzed, no species may be requested as output (and therefore, there is no ambiguity with regards to `concentration` vs. `amount`).


The format of the results file
------------------------------

The expected results stored in the file `NNNNN-results.csv` are simply organized as a table of values.  The values will be slightly different depending on whether a `TimeCourse` or `FluxBalanceSteadyState` test was requested.  When species are output, the values may represent either amounts or concentrations, depending on whether the species is listed in the `amount` or `concentration` field of the settings file.


### Results data format for TimeCourse tests

The expected results stored in the file `NNNNN-results.csv` are simply organized as a table of values.  The first column is simulation time, and the remaining columns are variables in the model (often species, but not necessarily – they could be compartments, parameters, or reaction rates too) in the same order as they are listed in the `variables`: line in the `NNNNN-settings.txt` file.  An optional header line is permitted at the top of the file. Here is a short example:

```
time,S1,S2
0,1.5e-15,0
0.1,1.357256127053693e-15,1.427438729463058e-16
0.2,1.228096128602129e-15,2.719038713978695e-16
0.3,1.111227330205311e-15,3.887726697946884e-16
0.4,1.005480064513687e-15,4.945199354863119e-16
0.5,1.005480064513687e-15,4.945199354863119e-16
```

The first line of the file lists the columns, and the rest are numerical data.  The total number of lines of data in the file is X+1, where X is the value of the `steps:` line in the `NNNNN-settings.txt` file.

It is possible for some values to be _Not a Number_ (indicating the result is not mathematically defined, such as attempting to divide by zero or perform an operation involving infinity).  It is also possible for values to be positive infinity or negative infinity.  There does not appear to be an agreed-upon standard way of expressing these values in comma-separated files, so the SBML Test Suite uses the following convention:

*  `NaN`  is the literal value used to indicate not-a-number
*  `INF`  is the literal value used to indicate positive infinity
*  `-INF` is the literal value used to indicate negative infinity

These symbols are treated in a case-insensitive manner by the SBML Test Suite.


## The results data format for FluxBalanceSteadyState tests

As is the case for the `TimeCourse` results, `FluxBalanceSteadyState` test results are stored in the file `NNNNN-results.csv`, and organized as a table of values.  The first, header line indicates the expected variables, and the second line indicates the value of those variables at steady state:

```
R01,R26,R10,R07,OBJF
1.0,1.0,0.5,0.5,1.0
```

It is possible for `FluxBalanceSteadyState` results to contain `Not a Number`, positive infinity and negative infinity.  These indicated using the symbols NAN, INF and -INF as used for the time-course test results.


Component tags and test tags
----------------------------

Tags are labels used to indicate properties of test cases in the SBML Test Suite.  The labels are used in the `NNNNN-model.m` files describing every test case.  Two different types of tags are used in the SBML Test Suite: component tags and test tags.  For each test case, the relevant tags are listed after the componentTags: and testTags: fields, respectively, in the `NNNNN-model.m` file.

* Component tags describe the SBML elements that are present in the model. Most models are tagged with more than one component tag. (Readers may wonder whether these tags could be inferred simply by parsing the SBML content, and the answer is yes, they could. The tags are provided so that software systems can determine which components are used in each model without having to parse the model first. )  Component tags are also present for SBML Level 3 package constructs.  Components from the Hierarchical Model Composition package are listed with a comp: prefix, and components from the Flux Balance Constraints package are listed with a fbc: prefix.

* Test tags may be combined in the same model. In fact, many of the SBML Test Suite test models are designed to explore escalating combinations of tests: one set of models may test a given tag, then another set of models may test that same tag plus a second tag, then a third model may test those two tags plus a third tag, in different combinations; and so on.  Some test tags indicate particular aspects of concepts introduced by a SBML Level 3 package. Tags that test aspects of the Hierarchical Model Composition package are prefixed with `comp:`, and tags that test aspects of the Flux Balance Constraints package are prefixed with `fbc:`.

There are many possible tags.  Please consult the following web page for a human-readable list and the definitions of each tag: [http://sbml.org/Software/SBML_Test_Suite/Case_Descriptions/Tags](http://sbml.org/Software/SBML_Test_Suite/Case_Descriptions/Tags)


Additional comments and tips
----------------------------

A frequent cause of difficulties in interpreting the SBML Test Suite results is that the output expected in the Test Suite results is not necessarily the same as the values for the species in the model.  In particular, the output values requested for some test cases may be in concentration units (alternatively, amounts) even if the species in the model have units of amounts (alternatively, concentrations).  For example, test cases 947 and 948 are the same as test cases 945 and 946, except that 945 and 946 ask for species `S1` output in amount units, while 947 and 948 ask for the output in concentration units.  (For the latter, you can see that the concentration suddenly drops by a factor of 10 when the compartment size is changed, as is correct when the amount stays the same.)  This is a purposeful test, to determine whether software tools can correctly handle SBML models that contain species expressed in either amounts or concentrations, and the possibility that compartment sizes may change over the course of simulations and alter one but not the other.


Copyright and license
---------------------

For full license information, please refer to the file [../LICENSE.txt](https://raw.githubusercontent.com/sbmlteam/moccasin/master/LICENSE.txt) in the parent directory.  Briefly, the test case distributions of the SBML Test Suite are distributed under the terms of the LGPL; the overall SBML Test Suite (including the software components) are distributed under the LGPL but include components from other sources licensed under other open-source terms.  (However, none of the terms are more restrictive than the LGPL.)


More information
----------------

More information about the SBML Test Suite is available online at
[http://sbml.org/Software/SBML_Test_Suite](http://sbml.org/Software/SBML_Test_Suite).

[![SBML Logo](http://sbml.org/images/8/82/Official-sbml-supported-70.jpg)](http://sbml.org)


The SBML Test Suite – Test Cases Archive
=========================================

Authors
-------

The SBML Test Suite has been in development for many years, and many people have contributed in various ways large and small.  The following are the primary authors of different parts:

* _Semantic test cases_:

  Sarah M. Keating<sup>a,b</sup>, Lucian P. Smith<sup>b,c</sup>, Bruce Shapiro <sup>b</sup>, Michael Hucka<sup>b</sup>, Frank T. Bergmann<sup>d</sup>, Brett Olivier<sup>g</sup>, Andrew Finney<sup>b</sup>

* _Stochastic test cases_:

  Thomas W. Evans<sup>e</sup>, Colin S. Gillespie<sup>f</sup>, Darren J. Wilkinson<sup>f</sup>, Lucian P. Smith<sup>b,c</sup>
   
* _Syntactic test cases_:

  Sarah M. Keating<sup>a,b</sup>, Lucian P. Smith<sup>b,c</sup>

Additional contributions are gratefully acknowledged from Ralph Gauges h</sup>, Chris Myers<sup>i</sup>, Akira Funahashi<sup>j</sup>, Andreas Dräger<sup>k</sup>, Roland Keller<sup>k</sup>, Fedor Kolpakov<sup>l</sup>, Stanley Gu<sup>c</sup>, and others in the SBML community.

Institutions:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>a</sup> EMBL-EBI, Hinxton, Cambridgeshire, UK<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>b</sup> California Institute of Technology, Pasadena, CA, US<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>c</sup> University of Washington, Seattle, WA, US<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>d</sup> University of Heidelberg, Heidelberg, DE<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>e</sup> University of Liverpool, Liverpool, UK<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>f</sup> Newcastle University, Newcastle, UK<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>g</sup> Vrije Universiteit Amsterdam<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>h</sup> Heidelberg Institute for Theoretical Studies, Heidelberg, DE<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>i</sup> University of Utah, Salt Lake City, UT, US<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>j</sup> Systems Biology Institute, Tokyo, JP<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>k</sup> University of Tuebingen, Tübingen, DE<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>l</sup> Institute of Systems Biology, Novosibirsk, RU   <br>


Introduction
------------

The SBML Test Suite test cases are divided into 3 sets.  Each set of cases tests a different aspect of [SBML](http://sbml.org/Documents/Specifications) understanding and compliance.

* **Semantic**: The semantic test suite contains valid SBML models with known, deterministic simulation results.  These models can be used to test the ability of a software system to understand the meaning of different SBML constructs and properly simulate the models in a deterministic fashion.  An example of a deterministic simulator is a system using a numerical differential-algebraic solver that supports discontinuous events.  Each test consists of a directory containing the model (or models, if that model can be translated to other SBML Levels + Versions combinations without semantic loss), together with instructions on how to simulate that model, and the expected results.  For more information about the semantic test cases, please look in the file named _README.md_ after you unzip the archive (i.e., the file _sbml-semantic-test-cases-YYYY-MM-DD.zip_, where _YYYY-MM-DD_ is the release date), or visit the [SBML Test Suite website](http://sbml.org/Software/SBML_Test_Suite).

* **Stochastic**: The stochastic test suite contains valid SBML models with known, stochastic simulation results.  These tests exercise fewer features of SBML, and are instead intended to test the accuracy of systems that employ discrete stochastic simulation algorithms.  Each test consists of a directory containing the model (or models, if that model can be translated to other SBML Levels + Versions combination without semantic loss), together with instructions on how to simulate that model, and the expected results.  Because the expected results are stochastic, each test case is designed to be performed multiple times, with summary statistics collected for each, which are then compared to the expected summary statistics using a formula derived from the number of times the test was repeated.  For more information about the semantic test cases, please look in the file named _README.md_ after you unzip the archive (i.e., the file _sbml-stochastic-test-cases-YYYY-MM-DD.zip_, where _YYYY-MM-DD_ is the release date), or visit the [SBML Test Suite website](http://sbml.org/Software/SBML_Test_Suite).

* **Syntactic**: The syntactic test suite consists of valid and invalid SBML models.  Each test is designed to check a particular SBML validation rule.  These rules are defined in the SBML specification documents.  Each case model in this part of the Test Suite is expected to be recognized by a software system as being either valid or invalid – nothing more.  The validity is indicated in the file name of a model.  (Example: _01002-fail-01-01-sev2-l2v1.txt_ means the test is expected to fail.)  Details about the error and the error message produced by libSBML are included, as are incidental warnings that libSBML may also produce for a given model/test case.  For more information about the semantic test cases, please look in the file named _README.md_ after you unzip the archive (i.e., the file _sbml-syntactic-test-cases-YYYY-MM-DD.zip_, where _YYYY-MM-DD_ is the release date), or visit the [SBML Test Suite website](http://sbml.org/Software/SBML_Test_Suite).


Usage
-----

For more information about using these test cases, please refer to the instructions either at the top level of the SBML Test Suite project repository, or in the individual subdirectories under this directory.


Licensing
---------

For full license information, please refer to the file [LICENSE.txt](LICENSE.txt).  Briefly, the test case distributions of the SBML Test Suite are distributed under the terms of the LGPL; the overall SBML Test Suite (including the software components) are distributed under the [LGPL](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html) but include components from other sources licensed under other open-source terms.  (However, none of the terms are more restrictive than the [LGPL](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html).)


Bugs and other issues
---------------------

Please report problems using the tracker at [https://github.com/sbmlteam/sbml-test-suite/issues](https://github.com/sbmlteam/sbml-test-suite/issues).


More information
----------------

More information about the SBML Test Suite is available online at
[http://sbml.org/Software/SBML_Test_Suite](http://sbml.org/Software/SBML_Test_Suite).

[![SBML Logo](https://raw.githubusercontent.com/sbmlteam/sbml-test-suite/develop/src/misc/graphics-originals/Official-sbml-supported-70.jpg)](http://sbml.org)


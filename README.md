# libRoadRunner
[![GitHub version](https://badge.fury.io/gh/sys-bio%2Froadrunner.svg)](http://badge.fury.io/gh/sys-bio%2Froadrunner)
[![Build Status](https://travis-ci.org/sys-bio/roadrunner.svg?branch=develop)](https://travis-ci.org/sys-bio/roadrunner)

Copyright 2013-2018

E. T. Somogyi <sup>1</sup>, J. K. Medley <sup>3</sup>, M. T. Karlsson <sup>2</sup>, M. Swat <sup>1</sup>, M. Galdzicki <sup>3</sup>, K. Choi <sup>3</sup>, W. Copeland <sup>3</sup> and H. M. Sauro <sup>3</sup>

1. Biocomplexity Institute, Indiana University, Simon Hall MSB1, Bloomington, IN 47405
2. Dune Scientific, 10522 Lake City Way NE, #302 Seattle WA
3. Department of Bioengineering, University of Washington, Seattle, WA, 98195

## Introduction

libRoadRunner is a high-performance and portable simulation engine for systems and synthetic biology.

## Contributing

**IMPORTANT!** Contributors **must** follow the [contribution guidelines](https://github.com/sys-bio/roadrunner/wiki). Contibuters are responsible for complying with the guidelines, including (but not limited to) making commits to the correct branch. Maintainers are not responsible for changes made to the wrong branch. Contributors take full responsibility for ensuring that their changes get merged into the develop branch.

## libRoadRunner supports the following features:

* Time Dependent Simulation (with optional conservation law reduction) using CVODE
* Supports SBML Level 2 to 3 but currently excludes algebraic rules and delay differential equations
* Uses latest libSBML distribution
* Defaults to LLVM code generation on the backend, resulting is very fast simulation times
* Optional generation of model C code and linking at run-time
* Add plugins, distribution comes with Levenberg-Marquardt optimizer plugin
* Compute steady state
* Metabolic Control Analysis
* Frequency Domain Analysis
* Access to:
  * Eigenvalues and Eigenvectors
  * Jacobian, full and reduced
  * Structural Matrices of the stoichiometry matrix


## Availability

RoadRunner is licensed for free as an open source programmatic library for use in other 
applications and as a standalone command line driven application. Its C++ API, C API, and 
Python APIs have comprehensive documentation. On Windows, OS X, and Linux binary files can be 

downloaded from http://sourceforge.net/projects/libroadrunner/files and can be installed 
ready for use.


## Acknowledgements

This work is funded by NIGMS grant: GM081070

Licence

Licensed under the Apache License, Version 2.0 (the License); you may not use this 
file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an ÎAS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS 
OF ANY KIND, either express or implied. See the License for the specific language 
governing permissions and limitations under the License.

In plain english this means:

You CAN freely download and use this software, in whole or in part, for personal, 
company internal, or commercial purposes;

You CAN use the software in packages or distributions that you create.

You SHOULD include a copy of the license in any redistribution you may make;

You are NOT required include the source of software, or of any modifications you may 
have made to it, in any redistribution you may assemble that includes it.

YOU CANNOT: redistribute any piece of this software without proper attribution;

 
  

libRoadRunner logo

  The libroadrunner logo is an adaptation of the image originally posted to Flickr by 
  El Brujo+ at http://flickr.com/photos/11039104@N08/2954808342. It was reviewed on 
  9 August 2009 by the FlickreviewR robot and was confirmed to be licensed under the 
  terms of the cc-by-sa-2.0.

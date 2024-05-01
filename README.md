# libRoadRunner
[![GitHub version](https://badge.fury.io/gh/sys-bio%2Froadrunner.svg)](http://badge.fury.io/gh/sys-bio%2Froadrunner)
[![Build Status](https://dev.azure.com/TheRoadrunnerProject/roadrunner/_apis/build/status/sys-bio.roadrunner?branchName=develop)](https://dev.azure.com/TheRoadrunnerProject/roadrunner/_build/latest?definitionId=8&branchName=develop)
[![Documentation Status](https://readthedocs.org/projects/libroadrunner/badge/?version=latest)](https://libroadrunner.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/libroadrunner.svg)](https://badge.fury.io/py/libroadrunner)

 <table style="width:100%">
  <tr>
    <td><img alt="Licence", src="https://img.shields.io/badge/License-Apache%202.0-yellowgreen"</td>
    <td><img alt="PyPI - Downloads", src="https://img.shields.io/pypi/dm/roadrunner"></td>
    <td><img alt="Funding", src="https://img.shields.io/badge/Funding-NIH%20(GM123032)-blue"></td>
    <td><img alt="Funding", src="https://img.shields.io/badge/Funding-NIBIB%20(EB028887)-blue"></td>    
   </tr>
</table> 

# Summary

libRoadRunner is a C/C++ library that supports simulation of SBML based models. It uses LLVM to generate extremely high performance code and is the fastest SBML-based simulator currently available (ref). Its main purpose is for use as a reusable library that can be hosted by other applications, particularly on large compute clusters for doing parameter optimization where performance is critical. It also has a set of python bindings that allow it to be easily used from python. 

We provide C/C++, Python and Julia bindings.

# Documentation

[Full Documentation](https://libroadrunner.readthedocs.io)

[Local Documentation](http://sys-bio.github.io/roadrunner/)  (May be used if above link is failing.)

[C API Documention](https://sys-bio.github.io/roadrunner/OriginalDoxygenStyleDocs/html/index.html)

# Installation

[Binaries](https://github.com/sys-bio/roadrunner/releases) for Windows, MacOS (including universal binaries for the M1 processor), and Linux

Python wheels are available at the above binaries link, and can also be installed via pip:

`pip install libroadrunner`

### Google Colab
libroadrunner can be used entirely in a browser using [Google Colab](https://colab.research.google.com/).  Ideally, it will work with the following steps (updated May 2023)
  
1. run a cell with ```!apt-get install libncurses5```
2. run a cell with ```!pip install libroadrunner```
3. (Vert important) Restart the runtime (From the menu: 'Runtime / Restart runtime')
4. Test by running ```import roadrunner```
  
The Python version behind Colab is reasonably stable but can change. This issue to changes in how colab handles packages. we try to keep abreast of changes at colab.

# Example Python usage

Once installed in Python (using pip or using the wheels directly), the following simple example script should demonstrate the basics of a roadrunner simulation:

```python
import roadrunner
rr = roadrunner.RoadRunner("https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000010.2?filename=BIOMD0000000010_url.xml")
results = rr.simulate(0, 2000, 200)
rr.plot()
print(results)
```

This simple script downloads [BioModels 10](https://www.ebi.ac.uk/biomodels/BIOMD0000000010) (the Kholodenko model of MAPK oscillation) and runs a simulation for 2000 seconds, storing the results in `results`.  If your system is set up to display figures, it will display a plot of the simulation, then print the numerical values obtained.

# Dependencies

All libRoadRunner binaries are self-contained, and should include all libraries it depends on.  When building libRoadRunner from source, its dependencies may either be downloaded *en masse* from the [libroadrunner-deps](https://github.com/sys-bio/libroadrunner-deps) repository, may be obtained directly from their [respective sources](https://github.com/sys-bio/libroadrunner-deps/tree/release/third_party), or the libraries may be installed directly on your operating system.  Compilation requires a C++17 compiler.

The one exception to the above is that the libRoadRunner binaries depend on the 'ncurses' library on linux.


# Copyright

Copyright 2013-2022

E. T. Somogyi <sup>1</sup>, J. K. Medley <sup>3</sup>, M. T. Karlsson <sup>2</sup>, M. Swat <sup>1</sup>, M. Galdzicki <sup>3</sup>, K. Choi <sup>3</sup>, W. Copeland <sup>3</sup>, L. Smith <sup>3</sup>, C. Welsh <sup>3</sup> and H. M. Sauro <sup>3</sup>

1. Biocomplexity Institute, Indiana University, Simon Hall MSB1, Bloomington, IN 47405
2. Dune Scientific, 10522 Lake City Way NE, #302 Seattle WA
3. Department of Bioengineering, University of Washington, Seattle, WA, 98195

The current (2021-present) developer is Lucian Smith.

## Contributing

Contributors may submit pull requests at any time, and are asked to follow some general [contribution guidelines](https://github.com/sys-bio/roadrunner/wiki).  Bug requests and feature suggestions are also welcomed on the [issues page](https://github.com/sys-bio/roadrunner/issues).

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
downloaded from the [Releases](https://github.com/sys-bio/roadrunner/releases) page, and the 
Python bindings can additionally be downloaded via pip. 


## Docker images
Currently we have a manylinux2014 [build](https://hub.docker.com/repository/docker/sysbiouw/roadrunner-manylinux2014-base) docker image. The 
base provides the environment you need to be able to build roadrunner yourself on manylinux2014 (centos 8).  

To get the base image: 

`docker pull sysbiouw/roadrunner-manylinux2014-base:llvm-13.x`

Docker build scripts can be found under the `docker` directory from the roadrunner
root directory. 

We can also build roadrunner in alternative docker environments (ubuntu etc.) on request. 

## Citing

If you use RoadRunner in your research, we would appreciate following citations in any works you publish:

Ciaran Welsh, Jin Xu, Lucian Smith, Matthias König, Kiri Choi, Herbert M Sauro, libRoadRunner 2.0: a high performance SBML simulation and analysis library, *Bioinformatics*, Volume 39, Issue 1, January 2023, btac770, https://doi.org/10.1093/bioinformatics/btac770

Endre T. Somogyi, Jean-Marie Bouteiller, James A. Glazier, Matthias König, J. Kyle Medley, Maciej H. Swat, Herbert M. Sauro, libRoadRunner: a high performance SBML simulation and analysis library, *Bioinformatics*, Volume 31, Issue 20, 15 October 2015, Pages 3315–3321, https://doi.org/10.1093/bioinformatics/btv363

## Acknowledgements

This work is funded by NIGMS grant: GM081070

## License

Licensed under the Apache License, Version 2.0 (the License); you may not use this 
file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an AS-IS BASIS, WITHOUT WARRANTIES OR CONDITIONS 
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

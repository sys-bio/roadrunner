#!/usr/bin/env python
##
## @file    qual_example1.py
## @brief   Qual Example
## @author  Sarah Keating
##
## <!--------------------------------------------------------------------------
## This sample program is distributed under a different license than the rest
## of libSBML.  This program uses the open-source MIT license, as follows:
##
## Copyright (c) 2013-2014 by the California Institute of Technology
## (California, USA), the European Bioinformatics Institute (EMBL-EBI, UK)
## and the University of Heidelberg (Germany), with support from the National
## Institutes of Health (USA) under grant R01GM070923.  All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
##
## Neither the name of the California Institute of Technology (Caltech), nor
## of the European Bioinformatics Institute (EMBL-EBI), nor of the University
## of Heidelberg, nor the names of any contributors, may be used to endorse
## or promote products derived from this software without specific prior
## written permission.
## ------------------------------------------------------------------------ -->

import sys
import os.path
from libsbml import *


def main (args):
    # Creates an SBMLNamespaces object with the given SBML level, version
    # package name, package version.
    sbmlns = SBMLNamespaces(3, 1, "qual", 1);

    # Creates an SBMLDocument object 
    document = SBMLDocument(sbmlns);

    # mark qual as required
    document.setPackageRequired("qual", True);

    # create the Model
    model = document.createModel();

    # create the Compartment
    compartment = model.createCompartment();
    compartment.setId("c");
    compartment.setConstant(True);
    
    # Get a QualModelPlugin object plugged in the model object.
    mplugin = model.getPlugin("qual");

    # create the QualitativeSpecies
    qs = mplugin.createQualitativeSpecies();
    qs.setId("s1");
    qs.setCompartment("c");
    qs.setConstant(False);
    qs.setInitialLevel(1);
    qs.setMaxLevel(4);
    qs.setName("sss");

    # create the Transition
    t = mplugin.createTransition();
    t.setId("d");
    t.setSBOTerm(1);

    i = t.createInput();
    i.setId("RD");
    i.setQualitativeSpecies("s1");
    i.setTransitionEffect("none");
    i.setSign("negative");
    i.setThresholdLevel(2);
    i.setName("aa");

    o = t.createOutput();
    o.setId("wd");
    o.setQualitativeSpecies("s1");
    o.setTransitionEffect("production");
    o.setOutputLevel(2);
    o.setName("aa");

    ft = t.createFunctionTerm();
    math = parseL3Formula("geq(s1, 2)");
    ft.setResultLevel(1);
    ft.setMath(math);

    dt = t.createDefaultTerm();
    dt.setResultLevel(2);

    writeSBML(document, "qual_example1.xml");

   

  
if __name__ == '__main__':
  main(sys.argv)  

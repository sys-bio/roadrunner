/**
 * @file    example1.cpp
 * @brief   SBML Qual example
 * @author  Sarah Keating
 *
 * <!--------------------------------------------------------------------------
 * This sample program is distributed under a different license than the rest
 * of libSBML.  This program uses the open-source MIT license, as follows:
 *
 * Copyright (c) 2013-2014 by the California Institute of Technology
 * (California, USA), the European Bioinformatics Institute (EMBL-EBI, UK)
 * and the University of Heidelberg (Germany), with support from the National
 * Institutes of Health (USA) under grant R01GM070923.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Neither the name of the California Institute of Technology (Caltech), nor
 * of the European Bioinformatics Institute (EMBL-EBI), nor of the University
 * of Heidelberg, nor the names of any contributors, may be used to endorse
 * or promote products derived from this software without specific prior
 * written permission.
 * ------------------------------------------------------------------------ -->
 */

#include <iostream>

#include "sbml/SBMLTypes.h"
#include "sbml/packages/qual/common/QualExtensionTypes.h"

LIBSBML_CPP_NAMESPACE_USE

int main(int argc,char** argv)
{

  if (argc != 2)
  {
    std::cout << "Usage: example1\n";
    return 1;
  }

  //
  // Creates an SBMLNamespaces object with the given SBML level, version
  // package name, package version.
  //
  // (NOTE) By defualt, the name of package (i.e. "qual") will be used
  // if the arugment for the prefix is missing or empty. Thus the argument
  // for the prefix can be added as follows:
  //
  //    SBMLNamespaces sbmlns(3,1,"qual",1,"QUAL");
  //

  SBMLNamespaces sbmlns(3,1,"qual",1);

  //
  // (NOTES) The above code creating an SBMLNamespaces object can be replaced 
  //         with one of the following other styles.
  //
  // (1) Creates an SBMLNamespace object with a SBML core namespace and then
  //     adds a qual package namespace to the object.
  //
  //         SBMLNamespaces sbmlns(3,1);
  //         sbmlns.addPkgNamespace("qual",1);
  //
  //          OR
  //
  //         SBMLNamespaces sbmlns(3,1);
  //         sbmlns.addNamespace(QualExtension::XmlnsL3V1V1,"qual");
  //
  // (2) Creates a QualPkgNamespaces object (SBMLNamespace derived class for
  //     qual package. The class is basically used for createing an SBase
  //     derived objects defined in the qual package) with the given SBML
  //     level, version, and package version
  //
  //        QualPkgNamespaces sbmlns(3,1,1);
  //     


  // create the document

  SBMLDocument *document = new SBMLDocument(&sbmlns);

  // mark qual as required

  document->setPackageRequired("qual", true);


  // create the Model

  Model* model=document->createModel();

  // create the Compartment

  Compartment* compartment = model->createCompartment();
  compartment->setId("c");
  compartment->setConstant(true);

  //
  // Get a QualModelPlugin object plugged in the model object.
  //
  // The type of the returned value of SBase::getPlugin() function is
  // SBasePlugin*, and thus the value needs to be casted for the
  // corresponding derived class.
  //
  QualModelPlugin* mplugin
    = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

  // create the QualitativeSpecies
  QualitativeSpecies* qs = mplugin->createQualitativeSpecies();
  qs->setId("s1");
  qs->setCompartment("c");
  qs->setConstant(false);
  qs->setInitialLevel(1);
  qs->setMaxLevel(4);
  qs->setName("sss");

  // create the Transition
  Transition* t = mplugin->createTransition();
  t->setId("d");
  t->setSBOTerm(1);

  Input* i = t->createInput();
  i->setId("RD");
  i->setQualitativeSpecies("s1");
  i->setTransitionEffect(INPUT_TRANSITION_EFFECT_NONE);
  i->setSign(INPUT_SIGN_NEGATIVE);
  i->setThresholdLevel(2);
  i->setName("aa");

  Output* o = t->createOutput();
  o->setId("wd");
  o->setQualitativeSpecies("s1");
  o->setTransitionEffect(OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  o->setOutputLevel(2);
  o->setName("aa");

  DefaultTerm* dt = t->createDefaultTerm();
  dt->setResultLevel(2) ;

  FunctionTerm* ft = t->createFunctionTerm();
  ASTNode* math = SBML_parseL3Formula("geq(s1, 2)");
  ft->setResultLevel(1);
  ft->setMath(math);
  
  writeSBML(document,"qual_example1.xml");
  delete document;

  return 0;
}

/**
 * @file    TestReadQualExtension.cpp
 * @brief   Unit tests of writing QualExtension 
 * @author  Akiya Jouraku
 *
 * $Id: $
 * $HeadURL: $
 *
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2013-2014 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *     3. University of Heidelberg, Heidelberg, Germany
 * 
 * Copyright (C) 2009-2011 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/


#include <limits>

#include <iostream>
#include <check.h>
#include <sbml/SBMLTypes.h>
#include <sbml/common/extern.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

extern char *TestDataDirectory;

START_TEST (test_QualExtension_read_L3V1V1)
{
  char *filename = safe_strcat(TestDataDirectory, "qual-example1.xml");
  SBMLDocument *document = readSBMLFromFile(filename);
  
  fail_unless(document->getPackageName() == "core");

  Model *model = document->getModel();

  fail_unless(model != NULL);
  fail_unless(model->getPackageName() == "core");

  // get the Qualitative species

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));
  fail_unless(mplugin != NULL);

  fail_unless(mplugin->getNumQualitativeSpecies() == 1);
  fail_unless(mplugin->getListOfQualitativeSpecies()->getPackageName() == "qual");

  fail_unless(mplugin->getNumTransitions() == 1);
  fail_unless(mplugin->getListOfTransitions()->getPackageName() == "qual");

  QualitativeSpecies* qs = mplugin->getQualitativeSpecies(0);
  fail_unless(qs->getPackageName() == "qual");
  fail_unless(qs->getId() == "s1");
  fail_unless(qs->getName() == "sss");
  fail_unless(qs->getCompartment() == "c");
  fail_unless(qs->getConstant() == false);
  fail_unless(qs->getInitialLevel() == 1);
  fail_unless(qs->getMaxLevel() == 4);

  Transition* t = mplugin->getTransition(0);

  
  fail_unless(t->getPackageName() == "qual");
  fail_unless(t->getId() == "d");
  fail_unless(t->getNumInputs() == 1);
  fail_unless(t->getNumOutputs() == 1);
  fail_unless(t->getNumFunctionTerms() == 1);
  fail_unless(t->isSetDefaultTerm() == true);

  Input *i = t->getInput(0);
  
  fail_unless(i->getPackageName() == "qual");
  fail_unless(i->getId() == "RD");
  fail_unless(i->getName() == "aa" );
  fail_unless(i->getQualitativeSpecies() == "s1");
  fail_unless(i->getTransitionEffect() == INPUT_TRANSITION_EFFECT_NONE );
  fail_unless(i->getSign() == INPUT_SIGN_NEGATIVE);
  fail_unless(i->getThresholdLevel() == 2);

  i = t->getInputBySpecies("s2");

  fail_unless(i == NULL);

  i = t->getInputBySpecies("s1");
  
  fail_unless(i->getPackageName() == "qual");
  fail_unless(i->getId() == "RD");
  fail_unless(i->getName() == "aa" );
  fail_unless(i->getQualitativeSpecies() == "s1");
  fail_unless(i->getTransitionEffect() == INPUT_TRANSITION_EFFECT_NONE );
  fail_unless(i->getSign() == INPUT_SIGN_NEGATIVE);
  fail_unless(i->getThresholdLevel() == 2);

  Output *o = t->getOutput(0);
  
  fail_unless(o->getPackageName() == "qual");
  fail_unless(o->getId() == "wd");
  fail_unless(o->getName() == "aa" );
  fail_unless(o->getQualitativeSpecies() == "s1");
  fail_unless(o->getTransitionEffect() == OUTPUT_TRANSITION_EFFECT_PRODUCTION );
  fail_unless(o->getOutputLevel() == 2);

  o = t->getOutputBySpecies("s2");

  fail_unless(o == NULL);

  o = t->getOutputBySpecies("s1");
  
  fail_unless(o->getPackageName() == "qual");
  fail_unless(o->getId() == "wd");
  fail_unless(o->getName() == "aa" );
  fail_unless(o->getQualitativeSpecies() == "s1");
  fail_unless(o->getTransitionEffect() == OUTPUT_TRANSITION_EFFECT_PRODUCTION );
  fail_unless(o->getOutputLevel() == 2);
  FunctionTerm* ft1 = t->getFunctionTerm(0);

  fail_unless(ft1->getPackageName() == "qual");
  fail_unless(ft1->getResultLevel() == 1);
  fail_unless(ft1->getMath() != NULL);
  fail_unless(ft1->isSetMath() == true);

  DefaultTerm* dt = t->getDefaultTerm();

  fail_unless(dt->getPackageName() == "qual");
  fail_unless(dt->getResultLevel() == 2);


  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_L3V1V1_defaultNS)
{
  char *filename = safe_strcat(TestDataDirectory, "qual-example1-defaultNS.xml");
  SBMLDocument *document = readSBMLFromFile(filename);
  
  fail_unless(document->getPackageName() == "core");

  Model *model = document->getModel();

  fail_unless(model != NULL);
  fail_unless(model->getPackageName() == "core");

  // get the Qualitative species

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));
  fail_unless(mplugin != NULL);

  fail_unless(mplugin->getNumQualitativeSpecies() == 1);
  fail_unless(mplugin->getListOfQualitativeSpecies()->getPackageName() == "qual");

  fail_unless(mplugin->getNumTransitions() == 1);
  fail_unless(mplugin->getListOfTransitions()->getPackageName() == "qual");

  QualitativeSpecies* qs = mplugin->getQualitativeSpecies(0);
  fail_unless(qs->getPackageName() == "qual");
  fail_unless(qs->getId() == "s1");
  fail_unless(qs->getName() == "sss");
  fail_unless(qs->getMetaId() == "_ddd");
  fail_unless(qs->getCompartment() == "c");
  fail_unless(qs->getConstant() == false);
  fail_unless(qs->getInitialLevel() == 1);
  fail_unless(qs->getMaxLevel() == 4);

  Transition* t = mplugin->getTransition(0);

  
  fail_unless(t->getPackageName() == "qual");
  fail_unless(t->getId() == "d");
  fail_unless(t->getNumInputs() == 1);
  fail_unless(t->getNumOutputs() == 1);
  fail_unless(t->getNumFunctionTerms() == 1);
  fail_unless(t->isSetDefaultTerm() == true);

  Input *i = t->getInput(0);
  
  fail_unless(i->getPackageName() == "qual");
  fail_unless(i->getId() == "RD");
  fail_unless(i->getName() == "aa" );
  fail_unless(i->getQualitativeSpecies() == "s1");
  fail_unless(i->getTransitionEffect() == INPUT_TRANSITION_EFFECT_NONE );
  fail_unless(i->getSign() == INPUT_SIGN_NEGATIVE);
  fail_unless(i->getThresholdLevel() == 2);

  Output *o = t->getOutput(0);
  
  fail_unless(o->getPackageName() == "qual");
  fail_unless(o->getId() == "wd");
  fail_unless(o->getName() == "asa" );
  fail_unless(o->getQualitativeSpecies() == "s1");
  fail_unless(o->getTransitionEffect() == OUTPUT_TRANSITION_EFFECT_PRODUCTION );
  fail_unless(o->getOutputLevel() == 2);

  FunctionTerm* ft1 = t->getFunctionTerm(0);

  fail_unless(ft1->getPackageName() == "qual");
  fail_unless(ft1->getResultLevel() == 1);
  fail_unless(ft1->getMath() != NULL);

  DefaultTerm* dt = t->getDefaultTerm();

  fail_unless(dt->getPackageName() == "qual");
  fail_unless(dt->getResultLevel() == 2);


  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_L3V1V1_unknown_elements)
{
  const char* s1 =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:qual=\"http://www.sbml.org/sbml/level3/version1/qual/version1\" level=\"3\" version=\"1\" qual:required=\"true\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <qual:listOfQualitativeSpecies>\n"
    "      <qual:qualitativeSpecies metaid=\"ddd\" qual:id=\"s1\" qual:compartment=\"c\" qual:constant=\"false\" qual:name=\"sss\" qual:initialLevel=\"1\" qual:maxLevel=\"4\"/>\n"
    "    </qual:listOfQualitativeSpecies>\n"
    "    <qual:listOfTransitions>\n"
    "      <qual:transition sboTerm=\"SBO:0000001\" qual:id=\"d\">\n"
    "        <qual:unknown>\n"
    "        </qual:unknown>\n"
    "        <qual:listOfInputs>\n"
    "          <qual:input qual:id=\"RD\" qual:qualitativeSpecies=\"s1\" qual:transitionEffect=\"none\" qual:name=\"aa\" qual:sign=\"negative\" qual:thresholdLevel=\"2\"/>\n"
    "        </qual:listOfInputs>\n"
    "        <qual:listOfOutputs>\n"
    "          <qual:output qual:id=\"wd\" qual:qualitativeSpecies=\"s1\" qual:transitionEffect=\"production\" qual:name=\"aa\" qual:outputLevel=\"2\"/>\n"
    "        </qual:listOfOutputs>\n"
    "        <qual:listOfFunctionTerms>\n"
		"          <qual:defaultTerm qual:resultLevel=\"1\"/>\n"
		"          <qual:functionTerm qual:resultLevel=\"2\">\n"
    "            <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
		"              <apply>\n"
		"                <geq/>\n"
		"                <ci> s1 </ci>\n"
		"                <cn type=\"integer\"> 2 </cn>\n"
		"              </apply>\n"
		"            </math>\n"
		"          </qual:functionTerm>\n"
		"        </qual:listOfFunctionTerms>\n"
    "      </qual:transition>\n"
    "    </qual:listOfTransitions>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  SBMLDocument *document = readSBMLFromString(s1);
  Model *model = document->getModel();
 
  fail_unless(model != NULL);
  fail_unless(document->getNumErrors() == 2);

  delete document;
}
END_TEST


Suite *
create_suite_ReadQualExtension (void)
{
  Suite *suite = suite_create("ReadQualExtension");
  TCase *tcase = tcase_create("ReadQualExtension");

  tcase_add_test( tcase, test_QualExtension_read_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_read_L3V1V1_defaultNS);
  tcase_add_test( tcase, test_QualExtension_read_L3V1V1_unknown_elements);
  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND

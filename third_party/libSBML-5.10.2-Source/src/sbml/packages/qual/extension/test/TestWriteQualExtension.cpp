/**
 * @file    TestWriteQualExtension.cpp
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
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/packages/qual/extension/QualExtension.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

static string QUAL_XMLNS_L3V1V1;
static QualExtension* G; 
static QualPkgNamespaces* GNS;

void
WriteQualExtensionTest_setup (void)
{
	try
	{
		G = new QualExtension();
		GNS = new QualPkgNamespaces();
		QUAL_XMLNS_L3V1V1 = GNS->getURI();
	}
	catch(...)
	{
		fail("Failed to create a QualExtension object");
	}
}


void
WriteQualExtensionTest_teardown (void)
{
	delete G;
	delete GNS;
}

START_TEST (test_QualExtension_create_and_write_L3V1V1)
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

  QualPkgNamespaces *sbmlns = new QualPkgNamespaces(3,1,1);

  // create the document

  SBMLDocument *document = new SBMLDocument(sbmlns);

  //// mark qual as required

  document->setPackageRequired("qual", true);

  // create the Model

  Model* model=document->createModel();

  // create the Compartment

  Compartment* compartment = model->createCompartment();
  compartment->setId("c");
  compartment->setConstant(true);

  //// create the QualitativeSpecies

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

  fail_unless(mplugin != NULL);

  QualitativeSpecies* qs = mplugin->createQualitativeSpecies();
  fail_unless(qs->setId("s1")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setCompartment("c")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setConstant(false)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setInitialLevel(1)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMaxLevel(4)            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setName("sss")            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMetaId("ddd")          == LIBSBML_OPERATION_SUCCESS);

  Transition* t = mplugin->createTransition();
  fail_unless(t->setId("d")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(t->setSBOTerm(1)            == LIBSBML_OPERATION_SUCCESS);

  Input* i = t->createInput();
  fail_unless(i->setId("RD")                 == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setQualitativeSpecies("s1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setTransitionEffect
  (InputTransitionEffect_fromString("none")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setSign(InputSign_fromString("negative"))
                                             == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setThresholdLevel(2)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setName("aa")               == LIBSBML_OPERATION_SUCCESS);

  Output* o = t->createOutput();
  fail_unless(o->setId("wd")                       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setQualitativeSpecies("s1")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setTransitionEffect
  (OutputTransitionEffect_fromString("production")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setOutputLevel(2)                 == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setName("aa")                     == LIBSBML_OPERATION_SUCCESS);

  DefaultTerm* dt = t->createDefaultTerm();
  fail_unless(dt->setResultLevel(1) == LIBSBML_OPERATION_SUCCESS);

  FunctionTerm* ft = t->createFunctionTerm();
  ASTNode* math = SBML_parseL3Formula("geq(s1, 2)");
  fail_unless(ft->setResultLevel(2) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(ft->setMath(math)     == LIBSBML_OPERATION_SUCCESS);


  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1,s2) == 0); 

  free(s2);

  // check clone()

  SBMLDocument* document2 = document->clone();
  s2 = writeSBMLToString(document2);
  fail_unless(strcmp(s1,s2) == 0); 
  free(s2);

  // check operator=

  Model *m = new Model(document->getSBMLNamespaces()); 
  m = document->getModel();
  document2->setModel(m);
  s2 = writeSBMLToString(document2);

  fail_unless(strcmp(s1,s2) == 0); 
  free(s2);
  delete document2;
 
  delete document;  
}
END_TEST


START_TEST (test_QualExtension_create_add_and_write_L3V1V1)
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

  QualPkgNamespaces *sbmlns = new QualPkgNamespaces(3,1,1);

  // create the document

  SBMLDocument *document = new SBMLDocument(sbmlns);

  //// mark qual as required

  document->setPackageRequired("qual", true);

  // create the Model

  Model* model= new Model(sbmlns);

  // create the Compartment

  Compartment* compartment = new Compartment(sbmlns);
  compartment->setId("c");
  compartment->setConstant(true);

  fail_unless(model->addCompartment(compartment) == LIBSBML_OPERATION_SUCCESS);

  //// create the QualitativeSpecies

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

  fail_unless(mplugin != NULL);

  QualitativeSpecies* qs = new QualitativeSpecies(sbmlns); 
  fail_unless(qs->setId("s1")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setCompartment("c")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setConstant(false)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setInitialLevel(1)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMaxLevel(4)            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setName("sss")            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMetaId("ddd")          == LIBSBML_OPERATION_SUCCESS);

  fail_unless(mplugin->addQualitativeSpecies(qs) == LIBSBML_OPERATION_SUCCESS);

  Transition* t = new Transition(sbmlns);
  fail_unless(t->setId("d")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(t->setSBOTerm(1)            == LIBSBML_OPERATION_SUCCESS);

  Input* i = new Input(sbmlns);
  fail_unless(i->setId("RD")                 == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setQualitativeSpecies("s1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setTransitionEffect
  (InputTransitionEffect_fromString("none")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setSign(InputSign_fromString("negative"))
                                             == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setThresholdLevel(2)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(i->setName("aa")               == LIBSBML_OPERATION_SUCCESS);

  fail_unless(t->addInput(i) == LIBSBML_OPERATION_SUCCESS);

  Output* o = new Output(sbmlns);
  fail_unless(o->setId("wd")                       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setQualitativeSpecies("s1")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setTransitionEffect
  (OutputTransitionEffect_fromString("production")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setOutputLevel(2)                 == LIBSBML_OPERATION_SUCCESS);
  fail_unless(o->setName("aa")                     == LIBSBML_OPERATION_SUCCESS);

  fail_unless(t->addOutput(o) == LIBSBML_OPERATION_SUCCESS);

  FunctionTerm* ft = new FunctionTerm(sbmlns);
  ASTNode* math = SBML_parseL3Formula("geq(s1, 2)");
  fail_unless(ft->setResultLevel(2) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(ft->setMath(math)     == LIBSBML_OPERATION_SUCCESS);

  fail_unless(t->addFunctionTerm(ft) == LIBSBML_OPERATION_SUCCESS);

  DefaultTerm* dt = new DefaultTerm(sbmlns);
  fail_unless(dt->setResultLevel(1) == LIBSBML_OPERATION_SUCCESS);


  fail_unless(t->setDefaultTerm(dt) == LIBSBML_OPERATION_SUCCESS);

  fail_unless(mplugin->addTransition(t) == LIBSBML_OPERATION_SUCCESS);

  fail_unless(document->setModel(model) == LIBSBML_OPERATION_SUCCESS);

  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1,s2) == 0); 

  free(s2);

  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_enable_via_model_and_write_L3V1V1)
{
  const char* s1 =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  const char* s1a =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:qual=\"http://www.sbml.org/sbml/level3/version1/qual/version1\" level=\"3\" version=\"1\" qual:required=\"true\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <qual:listOfQualitativeSpecies>\n"
    "      <qual:qualitativeSpecies qual:id=\"s1\" qual:compartment=\"c\" qual:constant=\"false\" qual:name=\"sss\" qual:initialLevel=\"1\" qual:maxLevel=\"4\"/>\n"
    "    </qual:listOfQualitativeSpecies>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  SBMLDocument *document = readSBMLFromString(s1);

  fail_unless(document->getNumPlugins()             == 0);

  Model *model = document->getModel();

  fail_unless(model != NULL);
  fail_unless(model->getNumPlugins() == 0);

  //
  // enable the qual package by invoking enablePackage function with Model object
  //
  fail_unless(model->enablePackage(QUAL_XMLNS_L3V1V1, "qual", true) == LIBSBML_OPERATION_SUCCESS);

  // mark qual as required

  document->setPackageRequired("qual", true);

  fail_unless(document->getNumPlugins() == 1);
  fail_unless(model->getNumPlugins()    == 1);

  //// create the QualitativeSpecies

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

  fail_unless(mplugin != NULL);

  QualitativeSpecies* qs = mplugin->createQualitativeSpecies();
  fail_unless(qs->setId("s1")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setCompartment("c")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setConstant(false)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setInitialLevel(1)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMaxLevel(4)            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setName("sss")            == LIBSBML_OPERATION_SUCCESS);

  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1a,s2) == 0); 
  
  free(s2);
  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_disable_via_model_and_write_L3V1V1)
{
  const char* s1 =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:qual=\"http://www.sbml.org/sbml/level3/version1/qual/version1\" level=\"3\" version=\"1\" qual:required=\"true\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <qual:listOfQualitativeSpecies>\n"
    "      <qual:qualitativeSpecies qual:id=\"s1\" qual:compartment=\"c\" qual:constant=\"false\" qual:name=\"sss\" qual:initialLevel=\"1\" qual:maxLevel=\"4\"/>\n"
    "    </qual:listOfQualitativeSpecies>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  const char* s1d =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  SBMLDocument *document = readSBMLFromString(s1);

  fail_unless(document->getNumPlugins() == 1);

  Model *model = document->getModel();

  fail_unless(model != NULL);
  fail_unless(model->getNumPlugins() == 1);

  //
  // disable the qual package by invoking enablePackage function with Model object
  //
  fail_unless(model->enablePackage(QUAL_XMLNS_L3V1V1, "qual", false) == LIBSBML_OPERATION_SUCCESS);

  fail_unless(document->getNumPlugins() == 0);
  fail_unless(model->getNumPlugins()    == 0);

  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1d,s2) == 0); 
  
  free(s2);
  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_enable_via_sbmldocument_and_write_L3V1V1)
{
  const char* s1 =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  const char* s1a =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:qual=\"http://www.sbml.org/sbml/level3/version1/qual/version1\" level=\"3\" version=\"1\" qual:required=\"true\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <qual:listOfQualitativeSpecies>\n"
    "      <qual:qualitativeSpecies qual:id=\"s1\" qual:compartment=\"c\" qual:constant=\"false\" qual:name=\"sss\" qual:initialLevel=\"1\" qual:maxLevel=\"4\"/>\n"
    "    </qual:listOfQualitativeSpecies>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  SBMLDocument *document = readSBMLFromString(s1);

  fail_unless(document->getNumPlugins()             == 0);

  //
  // enable the qual package by invoking enablePackage function with SBMLDocument object
  //
  fail_unless(document->enablePackage(QUAL_XMLNS_L3V1V1, "qual", true) == LIBSBML_OPERATION_SUCCESS);

  // mark qual as required

  document->setPackageRequired("qual", true);

  fail_unless(document->getNumPlugins()             == 1);

  Model *model = document->getModel();

  fail_unless(model != NULL);
  fail_unless(model->getNumPlugins() == 1);

  //// create the QualitativeSpecies

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

  fail_unless(mplugin != NULL);

  QualitativeSpecies* qs = mplugin->createQualitativeSpecies();
  fail_unless(qs->setId("s1")               == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setCompartment("c")       == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setConstant(false)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setInitialLevel(1)        == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setMaxLevel(4)            == LIBSBML_OPERATION_SUCCESS);
  fail_unless(qs->setName("sss")            == LIBSBML_OPERATION_SUCCESS);

  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1a,s2) == 0); 
  
  free(s2);
  delete document;  
}
END_TEST


START_TEST (test_QualExtension_read_disable_via_sbmldocument_and_write_L3V1V1)
{
  const char* s1 =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:qual=\"http://www.sbml.org/sbml/level3/version1/qual/version1\" level=\"3\" version=\"1\" qual:required=\"true\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <qual:listOfQualitativeSpecies>\n"
    "      <qual:qualitativeSpecies qual:id=\"s1\" qual:compartment=\"c\" qual:constant=\"false\" qual:name=\"sss\" qual:initialLevel=\"1\" qual:maxLevel=\"4\"/>\n"
    "    </qual:listOfQualitativeSpecies>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  const char* s1d =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model>\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"c\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "  </model>\n"
    "</sbml>\n"
    ;

  SBMLDocument *document = readSBMLFromString(s1);

  fail_unless(document->getNumPlugins() == 1);

  //
  // disable the qual package by invoking enablePackage function with Model object
  //
  fail_unless(document->enablePackage(QUAL_XMLNS_L3V1V1, "qual", false) == LIBSBML_OPERATION_SUCCESS);

  fail_unless(document->getNumPlugins() == 0);

  Model *model = document->getModel(); 

  fail_unless(model != NULL);
  fail_unless(model->getNumPlugins() == 0);

  char *s2 = writeSBMLToString(document);

  fail_unless(strcmp(s1d,s2) == 0); 
  
  free(s2);
  delete document;  
}
END_TEST



Suite *
create_suite_WriteQualExtension (void)
{
  Suite *suite = suite_create("WriteQualExtension");
  TCase *tcase = tcase_create("WriteQualExtension");

  tcase_add_checked_fixture(tcase, WriteQualExtensionTest_setup, WriteQualExtensionTest_teardown);	
	
  tcase_add_test( tcase, test_QualExtension_create_and_write_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_create_add_and_write_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_read_enable_via_model_and_write_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_read_disable_via_model_and_write_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_read_enable_via_sbmldocument_and_write_L3V1V1);
  tcase_add_test( tcase, test_QualExtension_read_disable_via_sbmldocument_and_write_L3V1V1);
  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND

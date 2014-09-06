/**
 * \file    TestExtensionObjects.cpp
 * \brief   Tests for the Comp extension objects
 * \author  Lucian Smith
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
 * Copyright 2011-2012 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <sbml/common/common.h>

#include <sbml/packages/comp/common/CompExtensionTypes.h>
#include <sbml/conversion/SBMLConverterRegistry.h>

#include <sbml/SBMLReader.h>
#include <sbml/SBMLTypes.h>

#include <string>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;

START_TEST (test_comp_enable_comp)
{
  const string filename(TestDataDirectory);
  const string cfile = filename + "nocomp.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());
  fail_unless(doc->getModel() != NULL);
  fail_unless(doc->getNumErrors(LIBSBML_SEV_ERROR) == 0);

  fail_unless(doc->isPackageEnabled("comp") == false);
  fail_unless(doc->getModel()->isPackageEnabled("comp") == false);
  fail_unless(doc->getModel()->getCompartment(0)->isPackageEnabled("comp") == false);

  int code = doc->enablePackage(CompExtension::getXmlnsL3V1V1(), "comp", true);
  fail_unless(code == LIBSBML_OPERATION_SUCCESS);
  doc->setPackageRequired("comp", true);

  fail_unless(doc->isPackageEnabled("comp") == true);
  fail_unless(doc->getModel()->isPackageEnabled("comp") == true);
  fail_unless(doc->getModel()->getCompartment(0)->isPackageEnabled("comp") == true);

}
END_TEST

START_TEST (test_comp_model)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  Model model(&sbmlns);
  CompModelPlugin* mplugin = static_cast<CompModelPlugin*>(model.getPlugin("comp"));

  fail_unless(mplugin->getNumSubmodels()==0);
  Submodel* submod = mplugin->createSubmodel();
  submod->setId("submod1");
  fail_unless(mplugin->getNumSubmodels()==1);
  fail_unless(mplugin->addSubmodel(NULL)==LIBSBML_INVALID_OBJECT);
  Submodel submod2(3, 1);
  submod2.setId("submod2");
  fail_unless(mplugin->addSubmodel(&submod2)==LIBSBML_INVALID_OBJECT);
  submod2.setModelRef("ID1");
  fail_unless(mplugin->addSubmodel(&submod2)==LIBSBML_OPERATION_SUCCESS);
  Submodel* submodref = mplugin->getSubmodel("submod2");
  fail_unless(submodref->getId()=="submod2");
  submodref = mplugin->getSubmodel(0);
  fail_unless(submodref->getId()=="submod1");
  submod->setModelRef("ID1");
  fail_unless(submodref->getModelRef()=="ID1");
  fail_unless(mplugin->removeSubmodel(3)==NULL);
  fail_unless(mplugin->removeSubmodel(0)==submod);
  fail_unless(mplugin->getSubmodel("submod1")==NULL);

  fail_unless(mplugin->getNumPorts()==0);
  Port* port = mplugin->createPort();
  port->setId("port1");
  fail_unless(mplugin->getNumPorts()==1);
  fail_unless(mplugin->addPort(NULL)==LIBSBML_INVALID_OBJECT);
  Port port2(3, 1);
  port2.setId("port2");
  fail_unless(mplugin->addPort(&port2)==LIBSBML_INVALID_OBJECT);
  port2.setIdRef("ID1");
  fail_unless(mplugin->addPort(&port2)==LIBSBML_OPERATION_SUCCESS);
  Port* portref = mplugin->getPort("port2");
  fail_unless(portref->getId()=="port2");
  portref = mplugin->getPort(0);
  fail_unless(portref->getId()=="port1");
  port->setIdRef("ID1");
  fail_unless(portref->getIdRef()=="ID1");
  fail_unless(mplugin->removePort(3)==NULL);
  fail_unless(mplugin->removePort(0)==port);
  fail_unless(mplugin->getPort("port1")==NULL);
}
END_TEST

START_TEST (test_comp_sbase)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  Parameter param(&sbmlns);
  CompSBasePlugin* pplugin = static_cast<CompSBasePlugin*>(param.getPlugin("comp"));

  fail_unless(pplugin->getNumReplacedElements()==0);
  ReplacedElement* re = pplugin->createReplacedElement();
  re->setMetaId("re1");
  fail_unless(pplugin->getNumReplacedElements()==1);
  fail_unless(pplugin->addReplacedElement(NULL)==LIBSBML_INVALID_OBJECT);
  ReplacedElement re2(3, 1);
  re2.setMetaId("re2");
  fail_unless(pplugin->addReplacedElement(&re2)==LIBSBML_INVALID_OBJECT);
  re2.setDeletion("ID1");
  fail_unless(pplugin->addReplacedElement(&re2)==LIBSBML_INVALID_OBJECT);
  re2.setSubmodelRef("mod1");
  fail_unless(pplugin->addReplacedElement(&re2)==LIBSBML_OPERATION_SUCCESS);
  ReplacedElement* reref = pplugin->getReplacedElement(1);
  fail_unless(reref != NULL);
  fail_unless(reref->getMetaId()=="re2");
  reref = pplugin->getReplacedElement(0);
  fail_unless(reref != NULL);
  fail_unless(reref->getMetaId()=="re1");
  re->setDeletion("ID1");
  fail_unless(reref->getDeletion()=="ID1");
  fail_unless(pplugin->removeReplacedElement(3)==NULL);
  fail_unless(pplugin->removeReplacedElement(0)==re);
  fail_unless(pplugin->getReplacedElement(1)==NULL);

  fail_unless(pplugin->isSetReplacedBy()==false);
  ReplacedBy* rb = pplugin->createReplacedBy();
  fail_unless(rb != NULL);
  fail_unless(pplugin->isSetReplacedBy()==true);
  fail_unless(pplugin->setReplacedBy(NULL)==LIBSBML_OPERATION_SUCCESS);
  fail_unless(pplugin->isSetReplacedBy()==false);
  ReplacedBy rb2(3,1);
  fail_unless(pplugin->setReplacedBy(&rb2)==LIBSBML_INVALID_OBJECT);
  rb2.setIdRef("ID1");
  fail_unless(pplugin->setReplacedBy(&rb2)==LIBSBML_INVALID_OBJECT);
  rb2.setSubmodelRef("mod1");
  fail_unless(pplugin->setReplacedBy(&rb2)==LIBSBML_OPERATION_SUCCESS);
  fail_unless(pplugin->unsetReplacedBy()==LIBSBML_OPERATION_SUCCESS);
}
END_TEST

START_TEST (test_comp_sbmldocument)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument doc(&sbmlns);
  CompSBMLDocumentPlugin* docplugin = static_cast<CompSBMLDocumentPlugin*>(doc.getPlugin("comp"));
  doc.setPackageRequired("comp", true);
  fail_unless(docplugin->getRequired() == true);

  fail_unless(docplugin->getNumModelDefinitions()==0);
  ModelDefinition* moddef = docplugin->createModelDefinition();
  moddef->setId("moddef1");
  fail_unless(docplugin->getNumModelDefinitions()==1);
  fail_unless(docplugin->addModelDefinition(NULL)==LIBSBML_INVALID_OBJECT);
  ModelDefinition moddef2(3, 1);
  moddef2.setId("moddef2");
  fail_unless(docplugin->addModelDefinition(&moddef2)==LIBSBML_OPERATION_SUCCESS);
  ModelDefinition* moddefref = docplugin->getModelDefinition("moddef2");
  fail_unless(moddefref->getId()=="moddef2");
  moddefref = docplugin->getModelDefinition(0);
  fail_unless(moddefref->getId()=="moddef1");
  moddef->setId("ID1");
  fail_unless(moddefref->getId()=="ID1");
  fail_unless(docplugin->removeModelDefinition(3)==NULL);
  fail_unless(docplugin->removeModelDefinition(0)==moddef);
  fail_unless(docplugin->getModelDefinition("moddef1")==NULL);

  fail_unless(docplugin->getNumExternalModelDefinitions()==0);
  ExternalModelDefinition* exmoddef = docplugin->createExternalModelDefinition();
  exmoddef->setId("exmoddef1");
  fail_unless(docplugin->getNumExternalModelDefinitions()==1);
  fail_unless(docplugin->addExternalModelDefinition(NULL)==LIBSBML_INVALID_OBJECT);
  ExternalModelDefinition exmoddef2(3, 1);
  exmoddef2.setId("exmoddef2");
  fail_unless(docplugin->addExternalModelDefinition(&exmoddef2)==LIBSBML_INVALID_OBJECT);
  exmoddef2.setSource("where/the/file/is.xml");
  fail_unless(docplugin->addExternalModelDefinition(&exmoddef2)==LIBSBML_OPERATION_SUCCESS);
  ExternalModelDefinition* exmoddefref = docplugin->getExternalModelDefinition("exmoddef2");
  fail_unless(exmoddefref->getId()=="exmoddef2");
  exmoddefref = docplugin->getExternalModelDefinition(0);
  fail_unless(exmoddefref->getId()=="exmoddef1");
  exmoddef->setModelRef("ID1");
  fail_unless(exmoddefref->getModelRef()=="ID1");
  fail_unless(docplugin->removeExternalModelDefinition(3)==NULL);
  fail_unless(docplugin->removeExternalModelDefinition(0)==exmoddef);
  fail_unless(docplugin->getExternalModelDefinition("exmoddef1")==NULL);

}
END_TEST

START_TEST (test_comp_model_flattening_with_ports)
{
  string filename(TestDataDirectory);
  string cfile = filename + "test14_A.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());
  Model* model = doc->getModel();
  fail_unless(model != NULL);


  // SK - since I made the flattenModel package protected
  // this will no longer work
  // ===========
  //CompModelPlugin* cmp = static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  //Model* flatmod = cmp->flattenModel();
  //fail_unless(flatmod != NULL);
  //SBMLNamespaces sbmlns(3,1,"comp",1);
  //SBMLDocument doc2(&sbmlns);
  //doc2.setPackageRequired("comp", true);
  //doc2.setModel(flatmod);
  //string newModel = writeSBMLToString(&doc2);
  //===========
  // replace with

  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("leavePorts", true);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  string newModel = writeSBMLToString(doc);

  string ffile = filename + "test14_A_flat_ports.xml";
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  fail_unless(flatModel == newModel);

  delete doc;
}
END_TEST

START_TEST (test_comp_flatten_exchange4)
{ 
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  // load document
  string cfile = filename + "exchangetest2_A.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  Model* model = doc->getModel();
  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(model != NULL);
  
  // SK - since I made the flattenModel package protected
  // this will no longer work
  // ===========
  //CompModelPlugin* cmp = static_cast<CompModelPlugin*>(model->getPlugin("comp"));

  //Model* flatmod = cmp->flattenModel();
  //fail_unless(flatmod != NULL);
  //CompPkgNamespaces csbmlns(3,1,1,"comp");
  //SBMLDocument flatdoc(&csbmlns);
  //flatdoc.setPackageRequired("comp", true);
  //flatdoc.setModel(flatmod);

  //string newModel = writeSBMLToString(&flatdoc);
  //===========
  // replace with

  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("leavePorts", true);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  string newModel = writeSBMLToString(doc);

  string ffile = filename + "exchangetest2_flat_ports.xml";
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  fail_unless(flatModel == newModel);

  delete doc;
  delete fdoc;
}
END_TEST
  
START_TEST (test_comp_flatten_exchange5)
{ 
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  // load document
  string cfile = filename + "CompTest_A.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  Model* model = doc->getModel();
  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(model != NULL);

  // SK - since I made the flattenModel package protected
  // this will no longer work
  // ===========
  //CompModelPlugin* cmp = static_cast<CompModelPlugin*>(model->getPlugin("comp"));

  //Model* flatmod = cmp->flattenModel();
  //fail_unless(flatmod != NULL);
  //CompPkgNamespaces csbmlns(3,1,1,"comp");
  //SBMLDocument flatdoc(&csbmlns);
  //flatdoc.setPackageRequired("comp", true);
  //flatdoc.setModel(flatmod);
  //writeSBMLToFile(&flatdoc, "CompTest_flat_ports.xml");

  //string newModel = writeSBMLToString(&flatdoc);
  //===========
  // replace with

  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("leavePorts", true);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  string newModel = writeSBMLToString(doc);

  string ffile = filename + "CompTest_flat_ports.xml";
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  fail_unless(flatModel == newModel);

  delete doc;
  delete fdoc;
}
END_TEST
  
Suite *
create_suite_TestExtensionObjects(void)
{ 
  TCase *tcase = tcase_create("SBMLCompExtensionObjects");
  Suite *suite = suite_create("SBMLCompExtensionObjects");

  tcase_add_test(tcase, test_comp_model);
  tcase_add_test(tcase, test_comp_sbase);
  tcase_add_test(tcase, test_comp_sbmldocument);
  tcase_add_test(tcase, test_comp_enable_comp);
  tcase_add_test(tcase, test_comp_model_flattening_with_ports);
  tcase_add_test(tcase, test_comp_flatten_exchange4);
  tcase_add_test(tcase, test_comp_flatten_exchange5);

  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


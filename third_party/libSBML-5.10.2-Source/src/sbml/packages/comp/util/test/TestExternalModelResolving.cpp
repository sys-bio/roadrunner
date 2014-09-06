/**
 * \file    TestExternalModelResolving.cpp
 * \brief   Implementation of the Tests for the URI resolvers
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

#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/validator/CompSBMLErrorTable.h>
#include <sbml/packages/comp/sbml/ExternalModelDefinition.h>
#include <sbml/SBMLReader.h>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;

START_TEST (test_comp_externalmodelresolving_)
{ 
  SBMLNamespaces sbmlns(3,1,"comp",1);
  CompPkgNamespaces csbmlns(3,1,1,"comp");

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  string fileuri(TestDataDirectory);
  fileuri = "file:" + fileuri + "test.xml";
  document->setLocationURI(fileuri);
  CompSBMLDocumentPlugin* compdoc = static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));

  ExternalModelDefinition* extdef = compdoc->createExternalModelDefinition();
  extdef->setSource("non://existing/uri");
  fail_unless(extdef->getReferencedModel() == NULL);

  fail_unless(document->getErrorLog()->contains(CompUnresolvedReference) == true);

  document->getErrorLog()->clearLog();

  extdef->setSource("enzyme_model.xml");
  Model* resolvedmod = extdef->getReferencedModel();
  fail_unless(resolvedmod != NULL);

  extdef->setModelRef("aggregate");
  Model* resolvedmod2 = extdef->getReferencedModel();
  fail_unless(resolvedmod == resolvedmod2);

  extdef->setModelRef("enzyme");
  resolvedmod2 = extdef->getReferencedModel();
  fail_unless(resolvedmod != resolvedmod2);
  fail_unless(resolvedmod2 != NULL);
  
  extdef->setModelRef("non_existent_id");
  resolvedmod2 = extdef->getReferencedModel();
  fail_unless(resolvedmod2 == NULL);

  fail_unless(document->getErrorLog()->contains(CompModReferenceMustIdOfModel) == true);

  document->getErrorLog()->clearLog();
  

 }
END_TEST


START_TEST (test_comp_externalmodelresolving_files)
{ 
  // load the document
  string fileuri(TestDataDirectory);
  fileuri = fileuri + "subdir/localExtMod.xml";
  SBMLDocument *document = readSBMLFromFile(fileuri.c_str());
  CompSBMLDocumentPlugin* compdoc = static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));

  ExternalModelDefinition* extdef = compdoc->getExternalModelDefinition(0);
  fail_unless(extdef != NULL);
  Model* resolvedmod = extdef->getReferencedModel();
  fail_unless(resolvedmod != NULL);

  delete document;
 }
END_TEST


Suite *
create_suite_TestExternalModelResolving (void)
{ 
  TCase *tcase = tcase_create("ExternalModelResolving");
  Suite *suite = suite_create("ExternalModelResolving");
  
  tcase_add_test(tcase, test_comp_externalmodelresolving_);
  tcase_add_test(tcase, test_comp_externalmodelresolving_files);
  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


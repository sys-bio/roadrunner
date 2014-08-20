/**
 * \file    TestModelFlattening.cpp
 * \brief   Tests for the various functions on CompModelPlugin that involve flattening and aggregation.
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

#ifdef LIBSBML_HAS_PACKAGE_LAYOUT
#include <sbml/packages/layout/common/LayoutExtensionTypes.h>
#endif

#include <sbml/SBMLReader.h>
#include <sbml/SBMLTypes.h>

#include <string>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;


START_TEST (test_modeldef_layout)
{
  string filename(TestDataDirectory);
  filename += "QTPop_flat_withlayout.xml";  
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());

  CompPkgNamespaces *ns = new CompPkgNamespaces(3);

  doc->enablePackage(ns->getURI(), "comp", true);
  doc->setPackageRequired("comp", true);

  // fail if document is null
  fail_unless(doc->getModel() != NULL);
  CompSBMLDocumentPlugin* csdp = static_cast<CompSBMLDocumentPlugin*>(doc->getPlugin("comp"));
  fail_unless(csdp != NULL);

  Model* model = doc->getModel();
  fail_unless(model != NULL);
  if (model==NULL) return;
  SBasePlugin* plugin = model->getPlugin("layout");
  if (plugin == NULL) 
  {
    // test does not apply
    return;
  }

  //CompModelPlugin *cmp = (CompModelPlugin *)model->getPlugin("comp");
  ModelDefinition* md = new ModelDefinition(*model);
  csdp->addModelDefinition(md);

  string result1 = writeSBMLToString(doc);

  SBMLDocument *doc2 = readSBMLFromString(result1.c_str());
  string result2 = writeSBMLToString(doc2);

  fail_unless(result1 == result2);

}
END_TEST
#ifdef LIBSBML_HAS_PACKAGE_LAYOUT

START_TEST (test_transfer_moddef)
{
  string filename(TestDataDirectory);
  filename += "ModelDefWithLayout.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  string origmodstr = writeSBMLToString(doc);
  fail_unless(origmodstr.find("layout:") != string::npos);
  CompSBMLDocumentPlugin* compdoc = static_cast<CompSBMLDocumentPlugin*>(doc->getPlugin("comp"));
  compdoc->setRequired(true);
  ModelDefinition* transfer = compdoc->getModelDefinition("QuorumTrigger");
  SBMLNamespaces sbmlns(3,1,"comp",1);
  LayoutPkgNamespaces layoutns(3, 1, 1, "layout");
  sbmlns.addNamespaces(layoutns.getNamespaces());
  SBMLDocument transferdoc(&sbmlns);
  transferdoc.createModel();
  compdoc = static_cast<CompSBMLDocumentPlugin*>(transferdoc.getPlugin("comp"));
  compdoc->setRequired(true);
  compdoc->addModelDefinition(transfer);
  SBMLDocumentPlugin* layoutdoc = static_cast<SBMLDocumentPlugin*>(transferdoc.getPlugin("layout"));
  layoutdoc->setRequired(false);
  string newmodstr = writeSBMLToString(&transferdoc);
  filename = TestDataDirectory;
  filename += "test_transfer_moddef.xml";
  doc = readSBMLFromFile(filename.c_str());
  string compare = writeSBMLToString(doc);
  fail_unless(newmodstr == compare);
}
END_TEST

START_TEST (test_transfer_moddef2mod)
{
  string filename(TestDataDirectory);
  filename += "ModelDefWithLayout.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  CompSBMLDocumentPlugin* compdoc = static_cast<CompSBMLDocumentPlugin*>(doc->getPlugin("comp"));
  fail_unless(compdoc != NULL);
  ModelDefinition* transfer = compdoc->getModelDefinition("QuorumTrigger");
  fail_unless(transfer != NULL);
  SBMLNamespaces sbmlns(3,1,"comp",1);
  LayoutPkgNamespaces layoutns(3, 1, 1, "layout");
  sbmlns.addNamespaces(layoutns.getNamespaces());
  SBMLDocument transferdoc(&sbmlns);
  transferdoc.setModel(transfer);
  SBMLDocumentPlugin* layoutdoc = static_cast<SBMLDocumentPlugin*>(transferdoc.getPlugin("layout"));
  layoutdoc->setRequired(false);
  compdoc = static_cast<CompSBMLDocumentPlugin*>(transferdoc.getPlugin("comp"));
  compdoc->setRequired(true);
  Model* newmod = transferdoc.getModel();
  fail_unless(newmod != NULL);
  string newmodstr = writeSBMLToString(&transferdoc);
  filename = TestDataDirectory;
  filename += "test_transfer_moddef2mod.xml";
  doc = readSBMLFromFile(filename.c_str());
  string compare = writeSBMLToString(doc);
  fail_unless(newmodstr == compare);
}
END_TEST
#endif

Suite *
create_suite_TestMultipleNamespaces(void)
{ 
  TCase *tcase = tcase_create("TestMultipleNamespaces");
  Suite *suite = suite_create("TestMultipleNamespaces");
  
  tcase_add_test(tcase, test_modeldef_layout);
#ifdef LIBSBML_HAS_PACKAGE_LAYOUT
  tcase_add_test(tcase, test_transfer_moddef);
  tcase_add_test(tcase, test_transfer_moddef2mod);
#endif
  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


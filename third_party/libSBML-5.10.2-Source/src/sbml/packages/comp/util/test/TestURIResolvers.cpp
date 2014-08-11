/**
 * \file    TestURIResolvers.cpp
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

#include <sbml/SBMLReader.h>
#include <sbml/SBMLDocument.h>
#include <sbml/packages/comp/util/SBMLUri.h>
#include <sbml/packages/comp/util/SBMLFileResolver.h>
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;
START_TEST (test_comp_sbmluri_parse)
{
  SBMLUri uri ("file:filename.txt");
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "filename.txt");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");
    
  uri  = "/path/to/filename.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "/path/to/filename.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri  = "relative/path/to/filename.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "relative/path/to/filename.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri  = "file:///C:/path/to/filename.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "C:/path/to/filename.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri  = "http://path/to/filename.xml?downloaddir=/here/it/is/#anchor";
  fail_unless(uri.getScheme() == "http");
  fail_unless(uri.getPath() == "to/filename.xml");
  fail_unless(uri.getQuery() == "downloaddir=/here/it/is/#anchor");
  fail_unless(uri.getHost() == "path");

  uri = "c:\\test.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "c:/test.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri = "http://identifiers.org/biomodels.db/BIOMD0000000010";
  fail_unless(uri.getScheme() == "http");
  fail_unless(uri.getPath() == "biomodels.db/BIOMD0000000010");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "identifiers.org");

  uri = "http://identifiers.org/biomodels.db/BIOMD0000000010?profile=most_reliable";
  fail_unless(uri.getScheme() == "http");
  fail_unless(uri.getPath() == "biomodels.db/BIOMD0000000010");
  fail_unless(uri.getQuery() == "profile=most_reliable");
  fail_unless(uri.getHost() == "identifiers.org");

  // i know the following is not strictly a url, but i believe we have
  // to support miriam urns for the time being
  uri = "urn:miriam:biomodels.db:BIOMD0000000005";
  fail_unless(uri.getScheme() == "urn:miriam:biomodels.db");
  fail_unless(uri.getPath() == "BIOMD0000000005");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri = SBMLUri("c:\\test").relativeTo("test.xml");
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "c:/test/test.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri = SBMLUri("http://identifiers.org/biomodels.db").relativeTo("BIOMD0000000010");
  fail_unless(uri.getScheme() == "http");
  fail_unless(uri.getPath() == "biomodels.db/BIOMD0000000010");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "identifiers.org");

  // the actual urls to be encountered will have this form *yikes*
  uri = "file:C:\\Development\\libsbml-with-packages\\src\\sbml\\packages\\comp\\util\\test/test-data/test.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "C:/Development/libsbml-with-packages/src/sbml/packages/comp/util/test/test-data/test.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  uri = "file:///C:\\Development\\libsbml-with-packages\\src\\sbml\\packages\\comp\\util\\test/test-data/test.xml";
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getPath() == "C:/Development/libsbml-with-packages/src/sbml/packages/comp/util/test/test-data/test.xml");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");

  // testing directory
  uri = TestDataDirectory;
  fail_unless(uri.getScheme() == "file");
  fail_unless(uri.getQuery() == "");
  fail_unless(uri.getHost() == "");
  
  SBMLUri uri2 = SBMLUri("").relativeTo(TestDataDirectory);
  fail_unless(uri2.getScheme() == "file");
  fail_unless(uri2.getPath() == uri.getPath());
  fail_unless(uri2.getQuery() == "");
  fail_unless(uri2.getHost() == "");
  
  // testing odd schemes
  string filename(TestDataDirectory);
  size_t slash = filename.find('\\');
  while (slash != string::npos) {
    filename.replace(slash, 1, "/");
    slash = filename.find('\\');
  }
  filename += "1090101-fail-01-04.xml";
  uri = SBMLUri("1//jjj/lllkl:jjkj");
  fail_unless(uri.getScheme() == "1//jjj/lllkl");
  fail_unless(uri.getPath() == "");
  uri = SBMLUri("1//jjj/lllkl:jjkj").relativeTo(filename);
  fail_unless(uri.getScheme() == "1//jjj/lllkl");
  fail_unless(uri.getPath() == filename);
  
  
}
END_TEST


  
START_TEST (test_comp_fileresolver_resolve)
{ 
  string filename(TestDataDirectory);
  filename += "complexified.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  fail_unless(doc->getLocationURI() == "file:" + filename);
  SBMLFileResolver fr;
  SBMLDocument* doc2 = fr.resolve("enzyme_model.xml", doc->getLocationURI());
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_1)
{ 
  string filename(TestDataDirectory);
  filename += "complexified.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  fail_unless(doc->getLocationURI() == "file:" + filename);
  SBMLFileResolver fr;
  string newFilename(TestDataDirectory);
  newFilename += "enzyme_model.xml";
  SBMLDocument* doc2 = fr.resolve(newFilename, doc->getLocationURI());
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_2)
{ 
  string filename(TestDataDirectory);
  filename += "complexified.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  fail_unless(doc->getLocationURI() == "file:" + filename);
  SBMLFileResolver fr;
  string newFilename("chrome://anything/enzyme_model.xml");
  SBMLDocument* doc2 = fr.resolve(newFilename, doc->getLocationURI());
  // this passes because we have inadvertently used
  // the file name at the end of the uri
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_3)
{ 
  string filename(TestDataDirectory);
  SBMLFileResolver fr;
  SBMLDocument* doc2 = fr.resolve("enzyme_model.xml", filename);
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_4)
{ 
  string filename(TestDataDirectory);
  filename += "complexified.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  fail_unless(doc->getLocationURI() == "file:" + filename);
  SBMLFileResolver fr;
  string newDir(TestDataDirectory);
  newDir += "/subdir";
  fr.addAdditionalDir(newDir);
  SBMLDocument* doc2 = fr.resolve("new_aggregate.xml", doc->getLocationURI());
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_5)
{ 
  SBMLFileResolver fr;
  string name("chrome://anything/enzyme_model.xml");
  string name1("http://anything/enzyme_model.xml");
  SBMLDocument* doc2 = fr.resolve(name, name1);
  // this passes because we have inadvertently used
  // the file nam at the end of the uri
  fail_unless(doc2 == NULL);
}
END_TEST
  
  
START_TEST (test_comp_fileresolver_resolve_6)
{ 
  string filename(TestDataDirectory);
  size_t pos = filename.rfind('/');
  if (pos != filename.npos)
  {
    filename = filename.substr(0, pos);
  }
  SBMLFileResolver fr;
  SBMLDocument* doc2 = fr.resolve("enzyme_model.xml", filename);
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
}
END_TEST
  
  
START_TEST (test_comp_resolverregistry_1)
{ 
  SBMLResolverRegistry &registry = SBMLResolverRegistry::getInstance();
  fail_unless(registry.getNumResolvers() == 1);
  SBMLFileResolver fr2;
  registry.addResolver(&fr2);
  fail_unless(registry.getNumResolvers()==2);
  registry.removeResolver(1);
  fail_unless(registry.getNumResolvers()==1);
}
END_TEST
  
  
START_TEST (test_comp_resolverregistry_2)
{ 
  const SBMLResolverRegistry &registry = SBMLResolverRegistry::getInstance();
  string filename(TestDataDirectory);
  filename += "complexified.xml";
  SBMLDocument* doc = readSBMLFromFile(filename.c_str());
  fail_unless(doc->getLocationURI() == "file:" + filename);
  SBMLDocument* doc2 = registry.resolve("enzyme_model.xml", doc->getLocationURI());
  fail_unless(doc2 != NULL);
  fail_unless(doc2->getModel() != NULL);
  doc2 = registry.resolve("non-existent-file.really");
  fail_unless(doc2 == NULL);
}
END_TEST
  
  
Suite *
create_suite_TestURIResolvers (void)
{ 
  TCase *tcase = tcase_create("SBMLCompURIResolvers");
  Suite *suite = suite_create("SBMLCompURIResolvers");
  
  tcase_add_test(tcase, test_comp_sbmluri_parse);
  tcase_add_test(tcase, test_comp_fileresolver_resolve);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_1);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_2);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_3);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_4);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_5);
  tcase_add_test(tcase, test_comp_fileresolver_resolve_6);
  tcase_add_test(tcase, test_comp_resolverregistry_1);
  tcase_add_test(tcase, test_comp_resolverregistry_2);
  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


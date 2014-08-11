/**
 * \file    TestCompFlatteningUnknownPackageRefs.cpp
 * \brief   Testing flatting models with unknown package references.
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

SBMLDocument* TestFlattenedUnknownNoValidate(string file1, string file2)
{
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", false);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + file1;  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  //For use in debugging the above statement.
  /*
  SBMLErrorLog* errors = doc->getErrorLog();
  if (errors->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) != 0) {
    fail_unless(false);
    for (unsigned long e=0; e<errors->getNumErrors(); e++) {
      const SBMLError* error = errors->getError(e);
      if (error->getSeverity() == LIBSBML_SEV_ERROR) {
        cout << error->getMessage() << endl;
      }
    }
  }
  */

  string newModel = writeSBMLToString(doc);
  //string outfile = filename + "unknown_flat.xml";
  //writeSBMLToFile(doc, outfile.c_str());
  string ffile = filename + file2;
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  fail_unless(flatModel == newModel);

  delete fdoc;
  delete converter;

  return doc;
}

SBMLDocument* TestFlattenedUnknownValidate(string file1, string file2)
{
  string filename(TestDataDirectory);
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + file1;  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  string newModel = writeSBMLToString(doc);

  string ffile = filename + file2;
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  
  fail_unless(flatModel == newModel);

  delete fdoc;
  delete converter;

  return doc;
}

SBMLDocument* TestFlattenedUnknownValidateFailsFlattening(string file1)
{
  string filename(TestDataDirectory);
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + file1;  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_FAILED);

  string newModel = writeSBMLToString(doc);

  SBMLDocument* fdoc = readSBMLFromFile(cfile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  
  fail_unless(flatModel == newModel);

  delete fdoc;
  delete converter;

  return doc;
}

SBMLDocument* TestFlattenedUnknownAbortNone(string file1, string file2)
{
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", false);
  props->addOption("abortIfUnflattenable", "none");

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + file1;  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  // fail if conversion was not valid
  fail_unless(result == LIBSBML_OPERATION_SUCCESS);

  //For use in debugging the above statement.
  /*
  SBMLErrorLog* errors = doc->getErrorLog();
  if (errors->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) != 0) {
    fail_unless(false);
    for (unsigned long e=0; e<errors->getNumErrors(); e++) {
      const SBMLError* error = errors->getError(e);
      if (error->getSeverity() == LIBSBML_SEV_ERROR) {
        cout << error->getMessage() << endl;
      }
    }
  }
  */

  string newModel = writeSBMLToString(doc);
  //string outfile = filename + "unknown_flat.xml";
  //writeSBMLToFile(doc, outfile.c_str());
  string ffile = filename + file2;
  SBMLDocument* fdoc = readSBMLFromFile(ffile.c_str());
  string flatModel = writeSBMLToString(fdoc);
  fail_unless(flatModel == newModel);

  delete fdoc;
  delete converter;

  return doc;
}


START_TEST (test_comp_flatten_unknown_1)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown1.xml", "unknown1_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_2)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown2.xml", "unknown2_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_3)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown3.xml", "unknown3_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_4)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown4.xml", "unknown4_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_5)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown5.xml", "unknown5_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_6)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown6.xml", "unknown6_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_7)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown7.xml", "unknown7_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_8)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown8.xml", "unknown8_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_9)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown9.xml", "unknown9_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_10)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown10.xml", "unknown10_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_11)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown11.xml", "unknown11_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_12)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown12.xml", "unknown12_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_13)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown13.xml", "unknown13_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_14)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown14.xml", "unknown14_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_15)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown15.xml", "unknown15_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_16)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown16.xml", "unknown16_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_17)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown17.xml", "unknown17_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_18)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown18.xml", "unknown18_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_19)
{ 
  SBMLDocument* doc = TestFlattenedUnknownNoValidate("unknown19.xml", "unknown19_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_20)
{ 
  SBMLDocument* doc = TestFlattenedUnknownAbortNone("unknown20.xml", "unknown20_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST


START_TEST (test_comp_flatten_unknown_21)
{ 
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", false);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + "flatten_fail_unknown.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  // write the doc before we attempt conversion
  string newModel = writeSBMLToString(doc);

  converter->setDocument(doc);
  int result = converter->convert();

  // we should fail because we are testing the restoreNamespaces function
  fail_unless(result == LIBSBML_OPERATION_FAILED);


  string flatModel = writeSBMLToString(doc);
  fail_unless(flatModel == newModel);

  delete converter; 

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  //fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;  
}
END_TEST


START_TEST (test_comp_flatten_unknown_withValidation_1)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown1.xml", "unknown1_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_2)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown2.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_3)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown3.xml", "unknown3_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_4)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown4.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_5)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown5.xml", "unknown5_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_6)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown6.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_7)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown7.xml", "unknown7_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_8)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown8.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 6);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_9)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown9.xml", "unknown9_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_10)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown10.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 7);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_11)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown11.xml", "unknown11_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_12)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown12.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_13)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown13.xml", "unknown13_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_14)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown14.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_15)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown15.xml", "unknown15_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 1);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_16)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown16.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_17)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown17.xml", "unknown17_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_18)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown18.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 6);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_19)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidate("unknown19.xml", "unknown19_flat.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(errors->getNumErrors() == 4);
  //fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);

  delete doc;  
}
END_TEST

START_TEST (test_comp_flatten_unknown_withValidation_20)
{ 
  SBMLDocument* doc = TestFlattenedUnknownValidateFailsFlattening("unknown20.xml");

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 7);
  fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  //fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;  
}
END_TEST


START_TEST (test_comp_flatten_unknown_withValidation_21)
{ 
  string filename(TestDataDirectory);
  //string filename("C:\\Development\\libsbml\\src\\sbml\\packages\\comp\\util\\test\\test-data\\");
  
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("basePath", filename);
  props->addOption("performValidation", true);

  SBMLConverter* converter = SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string cfile = filename + "flatten_fail_unknown.xml";  
  SBMLDocument* doc = readSBMLFromFile(cfile.c_str());

  // fail if there is no model (readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  // write the doc before we attempt conversion
  string newModel = writeSBMLToString(doc);

  converter->setDocument(doc);
  int result = converter->convert();

  // we should fail because file has errors that will halt flattening
  fail_unless(result == LIBSBML_OPERATION_FAILED);


  string flatModel = writeSBMLToString(doc);
  fail_unless(flatModel == newModel);

  delete converter; 

  SBMLErrorLog* errors = doc->getErrorLog();
  //fail_unless(errors->getNumErrors() == 5);
  //fail_unless(errors->contains(RequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedReqd) == true);
  //fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;  
}
END_TEST


Suite *
create_suite_TestFlatteningUnknownPackageRefs (void)
{ 
  TCase *tcase = tcase_create("SBMLCompFlatteningUnknownPackageRefs");
  Suite *suite = suite_create("SBMLCompFlatteningUnknownPackageRefs");
  
  tcase_add_test(tcase, test_comp_flatten_unknown_1);
  tcase_add_test(tcase, test_comp_flatten_unknown_2);
  tcase_add_test(tcase, test_comp_flatten_unknown_3);
  tcase_add_test(tcase, test_comp_flatten_unknown_4);
  tcase_add_test(tcase, test_comp_flatten_unknown_5);
  tcase_add_test(tcase, test_comp_flatten_unknown_6);
  tcase_add_test(tcase, test_comp_flatten_unknown_7);
  tcase_add_test(tcase, test_comp_flatten_unknown_8);
  tcase_add_test(tcase, test_comp_flatten_unknown_9);
  tcase_add_test(tcase, test_comp_flatten_unknown_10);
  tcase_add_test(tcase, test_comp_flatten_unknown_11);
  tcase_add_test(tcase, test_comp_flatten_unknown_12);
  tcase_add_test(tcase, test_comp_flatten_unknown_13);
  tcase_add_test(tcase, test_comp_flatten_unknown_14);
  tcase_add_test(tcase, test_comp_flatten_unknown_15);
  tcase_add_test(tcase, test_comp_flatten_unknown_16);
  tcase_add_test(tcase, test_comp_flatten_unknown_17);
  tcase_add_test(tcase, test_comp_flatten_unknown_18);
  tcase_add_test(tcase, test_comp_flatten_unknown_19);
  tcase_add_test(tcase, test_comp_flatten_unknown_20);
  tcase_add_test(tcase, test_comp_flatten_unknown_21);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_1);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_2);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_3);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_4);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_5);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_6);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_7);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_8);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_9);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_10);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_11);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_12);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_13);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_14);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_15);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_16);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_17);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_18);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_19);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_20);
  tcase_add_test(tcase, test_comp_flatten_unknown_withValidation_21);
 
  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


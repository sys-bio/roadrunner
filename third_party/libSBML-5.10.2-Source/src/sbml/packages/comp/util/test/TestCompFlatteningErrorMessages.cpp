/**
 * \file    TestCompFlatteningErrorMessages.cpp
 * \brief   Implementation of the Tests for the Comp flattening converter
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

START_TEST(test_comp_flatten_invalid)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "1020616-fail-01-01.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_CONV_INVALID_SRC_DOCUMENT);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid2)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  CompModelPlugin* mplugin = 
           static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an incorrect external model definition.
  //ExternalModelDefinition* emd =
  compdoc->createExternalModelDefinition();

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid3)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an incorrect external model definition.
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompExtModDefAllowedAttributes) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid4)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
            static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  CompModelPlugin* mplugin = 
            static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an incorrect external model definition.
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  emd->setSource("nonexistent:x");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid5)
{
  //In this version the validation *is* performed before running.  
  // This tests the fix I made to 
  // ExternalModelReferenceCycles 
  // so it doesn't expect a 5-character URI prefix.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", true);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an incorrect external model definition.
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  emd->setSource("nonexistent:x");

  document->setLocationURI("nonexistent2:y");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_CONV_INVALID_SRC_DOCUMENT);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid6)
{
  //In this version the validation *is* performed before running.  
  // This tests the fix I made to 
  // ExternalModelReferenceCycles so it doesn't expect any 
  // locationURI to be set at all.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", true);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
    static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an incorrect external model definition.
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  emd->setSource("nonexistent:x");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_CONV_INVALID_SRC_DOCUMENT);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid7)
{
  //Test recursive submodels.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("Mod1");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompSubmodelCannotReferenceSelf) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid8)
{
  //Test recursive submodels.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submodA");
  submod1->setModelRef("Mod1");

  // create a model definition
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("Mod1");
  mplugin = static_cast<CompModelPlugin*>(moddef->getPlugin("comp"));

  // create a recursive Submodel
  submod1 = mplugin->createSubmodel();
  submod1->setId("submodB");
  submod1->setModelRef("mainmod");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompSubmodelCannotReferenceSelf) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid9)
{
  //Test submodels with no ID.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
        static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel with no ID.
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setModelRef("Mod1");

  // create a model definition
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("Mod1");
  mplugin = static_cast<CompModelPlugin*>(moddef->getPlugin("comp"));

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompSubmodelAllowedAttributes) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid10)
{
  //Test submodels with no modelRef.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel with no modelRef.
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompSubmodelAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid11)
{
  //Test submodels with incorrect modelRef.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
    static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel with incorrect modelRef.
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid12)
{
  //Test external model definition that points to an l2 model.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition that points to an l2 model.
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/level2.xml";
  emd->setSource(filename);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReferenceMustBeL3) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid13)
{
  //Test external model definition that points to a directory 
  // (a real thing, but something libsbml can't open)
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
          static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition that points to a directory
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  emd->setSource(filename);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid14)
{
  //Test external model definition that points to a text file
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition that points to a text file
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/model.txt";
  emd->setSource(filename);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoModelInReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid15)
{
  //Test external model definition that points to an empty L3 model
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/empty.xml";
  emd->setSource(filename);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoModelInReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid16)
{
  //Test external model definition with a modelRef
  // that points to an empty L3 model
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/empty.xml";
  emd->setSource(filename);
  emd->setModelRef("model");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoModelInReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid17)
{
  //Test external model definition with a modelRef
  //that points to an empty L3 model
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/empty-comp.xml";
  emd->setSource(filename);
  emd->setModelRef("model");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoModelInReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid18)
{
  //Test external model definition with a modelRef doesn't 
  //appear in the referenced L3 document.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/new_aggregate.xml";
  emd->setSource(filename);
  emd->setModelRef("model");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompModReferenceMustIdOfModel) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid19)
{
  //Test external model definition with a modelRef doesn't 
  //appear in the referenced L3 document.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/onemod.xml";
  emd->setSource(filename);
  emd->setModelRef("model");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompModReferenceMustIdOfModel) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid20)
{
  //Test external model definition with a modelRef doesn't 
  //appear in the referenced L3 document.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
         static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                  static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/onemodnamed.xml";
  emd->setSource(filename);
  emd->setModelRef("model");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompModReferenceMustIdOfModel) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid21)
{
  //Test external model definition with a modelRef doesn't appear in the referenced L3 document.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  //int retval = 0;
  int rv;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/badextmod.xml";
  emd->setSource(filename);
  emd->setModelRef("bad");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid22)
{
  //Test external model definition from a document with comp turned off.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create an external model definition
  ExternalModelDefinition* emd = compdoc->createExternalModelDefinition();
  emd->setId("Mod1");
  string filename(TestDataDirectory);
  filename += "subdir/badextmod.xml";
  emd->setSource(filename);
  emd->setModelRef("bad");

  //Now try to turn off comp in the main document
  document->disablePackage(submod1->getURI(), "comp");

  //Now call 'getReferencedModel' directly, 
  // since that's the only way to trigger this particular error message.
  Model* mod = emd->getReferencedModel();
  fail_unless(mod == NULL);

  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompUnresolvedReference) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid23)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  

  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  //ReplacedElement* re =
  pcomp->createReplacedElement();
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReplacedElementAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid24)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  

  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                 static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReplacedElementAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid25)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                    static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompIdRefMustReferenceObject) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid26)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setMetaIdRef("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompMetaIdRefMustReferenceObject) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid27)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setPortRef("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompPortRefMustReferencePort) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid28)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setUnitRef("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompUnitRefMustReferenceUnitDef) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid29)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompDeletionMustReferenceObject) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid30)
{
  //Test replacedBy without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedBy* rb = pcomp->createReplacedBy();
  rb->setSubmodelRef("submod1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReplacedByAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid31)
{
  //Test port without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a port
  //Port* port =
  mplugin->createPort();

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompPortAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid32)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // create a Deletion
  //Deletion* del =
  submod1->createDeletion();

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompDeletionAllowedAttributes) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid33)
{
  //Test SBaseRef without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element that points to a submodel...
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("submod2");

  //And a child SBaseRef that points to nothing.
  //SBaseRef* sbr =
  re->createSBaseRef();
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition with a submodel.
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");
  mplugin = static_cast<CompModelPlugin*>(md->getPlugin("comp"));
  submod1 = mplugin->createSubmodel();
  submod1->setId("submod2");
  submod1->setModelRef("Mod2");


  md = compdoc->createModelDefinition();
  md->setId("Mod2");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompSBaseRefMustReferenceObject) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid34)
{
  //Test SBaseRef that shouldn't be there.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element that points to a parameter...
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("p2");

  //And a child SBaseRef that points to nothing.
  //SBaseRef* sbr =
  re->createSBaseRef();
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition with a submodel.
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");
  param = md->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompParentOfSBRefChildMustBeSubmodel) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid35)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  SBaseRef sbr(3, 1, 1);
  sbr.setSBMLDocument(&document);
  sbr.saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid36)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  Parameter param(&sbmlns);
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param.getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSBMLDocument(&document);
  re->setSubmodelRef("sub1");
  re->setIdRef("p1");
  re->saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid37)
{
  //Test ReplacedElement with errant submodelRef.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  //CompModelPlugin* mplugin =
  //                 static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReplacedElementSubModelRef) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid38)
{
  //Test ReplacedBy with errant submodelRef.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  //CompModelPlugin* mplugin =
  //                 static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced by element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedBy* re = pcomp->createReplacedBy();
  re->setSubmodelRef("submod1");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompReplacedBySubModelRef) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid39)
{
  //Test mismatched IDs in replacement.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("p2");

  //Create a submodel
  Submodel* submod = mplugin->createSubmodel();
  submod->setId("submod1");
  submod->setModelRef("moddef1");

  //Create a model definition with a parameter.
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("moddef1");
  param = moddef->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompMustReplaceIDs) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid40)
{
  //Test mismatched metaIDs in replacement.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setConstant(true);
  param->setId("p1");

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("p2");

  //Create a submodel
  Submodel* submod = mplugin->createSubmodel();
  submod->setId("submod1");
  submod->setModelRef("moddef1");

  //Create a model definition with a parameter.
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("moddef1");
  param = moddef->createParameter();
  param->setId("p2");
  param->setMetaId("p2_meta");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompMustReplaceMetaIDs) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid41)
{
  //Test mismatched IDs in replacement.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setConstant(true);
  param->setId("p1");

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedBy* re = pcomp->createReplacedBy();
  re->setSubmodelRef("submod1");
  re->setMetaIdRef("p2");

  //Create a submodel
  Submodel* submod = mplugin->createSubmodel();
  submod->setId("submod1");
  submod->setModelRef("moddef1");

  //Create a model definition with a parameter.
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("moddef1");
  param = moddef->createParameter();
  param->setMetaId("p2");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompMustReplaceIDs) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid42)
{
  //Test mismatched metaIDs in replacement.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setConstant(true);
  param->setId("p1");
  param->setMetaId("p1_meta");

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedBy* re = pcomp->createReplacedBy();
  re->setSubmodelRef("submod1");
  re->setIdRef("p2");

  //Create a submodel
  Submodel* submod = mplugin->createSubmodel();
  submod->setId("submod1");
  submod->setModelRef("moddef1");

  //Create a model definition with a parameter.
  ModelDefinition* moddef = compdoc->createModelDefinition();
  moddef->setId("moddef1");
  param = moddef->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompMustReplaceMetaIDs) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid43)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ReplacedBy rb(3, 1, 1);
  rb.setSBMLDocument(&document);
  rb.performReplacement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid44)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ReplacedElement re(3, 1, 1);
  re.setSBMLDocument(&document);
  re.setSubmodelRef("submod1");
  re.performReplacement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid45)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOfReplacedElements lore(3, 1, 1);
  ReplacedElement* re = new ReplacedElement(3, 1, 1);
  lore.appendAndOwn(re);
  re->setSBMLDocument(&document);
  re->setSubmodelRef("submod1");
  re->performReplacement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid46)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOf lore(3, 1);
  ReplacedElement* re = new ReplacedElement(3, 1, 1);
  lore.appendAndOwn(re);
  re->setSBMLDocument(&document);
  re->setSubmodelRef("submod1");
  re->performReplacement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid47)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ReplacedElement re(3, 1, 1);
  re.setSBMLDocument(&document);
  re.setSubmodelRef("submod1");
  re.setDeletion("del1");
  re.getReferencedElementFrom(NULL);

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid48)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  Model* mod = document.createModel();
  Parameter* param = mod->createParameter();
  CompSBasePlugin* paramcomp = 
                  static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = paramcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  re->getReferencedElementFrom(NULL);

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompReplacedElementSubModelRef) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid49)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOf lo(3, 1);
  SBaseRef* sbr = new SBaseRef(3, 1, 1);
  lo.appendAndOwn(sbr);
  sbr->setSBMLDocument(&document);
  sbr->saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid50)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOf lo(3, 1);
  Deletion* deletion = new Deletion(3, 1, 1);
  lo.appendAndOwn(deletion);
  deletion->setSBMLDocument(&document);
  deletion->saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid51)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  Deletion del(3, 1, 1);
  del.setSBMLDocument(&document);
  del.saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid52)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOfDeletions lo(3, 1, 1);
  Deletion* deletion = new Deletion(3, 1, 1);
  lo.appendAndOwn(deletion);
  deletion->setSBMLDocument(&document);
  deletion->saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid53)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  ListOfDeletions* lodels = new ListOfDeletions(3, 1, 1);
  Deletion* deletion = new Deletion(3, 1, 1);
  lodels->appendAndOwn(deletion);
  ListOf lo(3,1);
  lo.appendAndOwn(lodels);

  deletion->setSBMLDocument(&document);
  deletion->saveReferencedElement();
  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid54)
{
  SBMLNamespaces sbmlns(3,1,"comp",1);
  SBMLDocument document(&sbmlns);
  Port port(3, 1, 1);
  port.setSBMLDocument(&document);
  port.saveReferencedElement();

  SBMLErrorLog* errors = document.getErrorLog();

  fail_unless(errors->getNumErrors() == 1);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
}
END_TEST

START_TEST(test_comp_flatten_invalid55)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "1020310-fail-01-01.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompSubmodelCannotReferenceSelf) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid56)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "1020310-fail-01-02.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompSubmodelCannotReferenceSelf) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid57)
{
  //Test double deletion.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // create a deletion
  Deletion* del = submod1->createDeletion();
  del->setIdRef("p1");
  del = submod1->createDeletion();
  del->setIdRef("p1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  //With a parameter
  Parameter* param = md->createParameter();
  param->setId("p1");
  param->setConstant("true");

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid58)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "multiref1.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();
  fail_unless(result==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid59)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "multiref2.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();
  fail_unless(result==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid60)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "multiref3.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();
  fail_unless(result==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid61)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("sp1");

  //create a second parameter with a replacedBy pointing to the same thing
  param = model->createParameter();
  param->setId("p2");
  param->setConstant(true);
  pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedBy* rb = pcomp->createReplacedBy();
  rb->setSubmodelRef("submod1");
  rb->setIdRef("sp1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  param = md->createParameter();
  param->setId("sp1");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid62)
{
  //Test replaced element without correct attributes.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel with a deletion
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");
  Deletion* del = submod1->createDeletion();
  del->setId("del1");
  del->setIdRef("sp1");

  //Create a parameter that points to the deletion
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);
  CompSBasePlugin* cparam = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = cparam->createReplacedElement();
  re->setDeletion("del1");
  re->setSubmodelRef("submod1");

  //And a port that also points to the deletion
  Port* port = mplugin->createPort();
  port->setId("port1");
  port->setIdRef("del1");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  param = md->createParameter();
  param->setId("sp1");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid63)
{
  //Test invalid flattened models.
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", true);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel with a deletion
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");
  Deletion* del = submod1->createDeletion();
  del->setId("del1");
  del->setIdRef("compartment");

  // Create a model definition
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");

  Species* species = md->createSpecies();
  species->setId("S1");
  species->setConstant(true);
  species->setCompartment("compartment");

  Compartment* comp = md->createCompartment();
  comp->setId("compartment");
  comp->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_CONV_INVALID_SRC_DOCUMENT);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompFlatModelNotValid) == true);
  fail_unless(errors->contains(NeedCompartmentIfHaveSpecies) == true);
  fail_unless(errors->contains(InvalidSpeciesCompartmentRef) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid64)
{
  //Test invalid flattened models.
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "flatten_fail1.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_CONV_INVALID_SRC_DOCUMENT);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompFlatModelNotValid) == true);
  fail_unless(errors->contains(ApplyCiMustBeModelComponent) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid65)
{
  //Test invalid flattened models.
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "flatten_fail2.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_CONV_INVALID_SRC_DOCUMENT);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompFlatModelNotValid) == true);
  fail_unless(errors->contains(InvalidSpeciesReference) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid_core)
{
  //Test invalid flattened models.
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "invalid_core_fail1.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_CONV_INVALID_SRC_DOCUMENT);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) == 1);
  fail_unless(errors->contains(AssignmentToConstantEntity) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid66)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "self-ref.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompCircularExternalModelReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);
  
  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid67)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "self-ref2.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompCircularExternalModelReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid68)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "self-ref3.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompCircularExternalModelReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid69)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "self-ref4.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompCircularExternalModelReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid70)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "self-ref5.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_OPERATION_FAILED);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompCircularExternalModelReference) == true);
  fail_unless(errors->contains(CompSubmodelMustReferenceModel) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid71)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create another Parameter in it.
  param = model->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Give it a replaced element
  pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create a Port for the deletion (this is the problem)
  Port* port = mplugin->createPort();
  port->setId("port1");
  port->setIdRef("del1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("Mod1");

  // give it a deletion
  Deletion* del = submod1->createDeletion();
  del->setIdRef("deleteme");
  del->setId("del1");

  // Create a model definition with a parameter to be deleted
  ModelDefinition* md = compdoc->createModelDefinition();
  md->setId("Mod1");
  param = md->createParameter();
  param->setId("deleteme");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid72)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = 
                   static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setIdRef("del1");
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("moddef1");

  // create a Model Definition
  ModelDefinition* md=compdoc->createModelDefinition();
  md->setId("moddef1");
  mplugin = static_cast<CompModelPlugin*>(md->getPlugin("comp"));
  
  // create a Parameter in it.
  param = md->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create another Parameter in it.
  param = md->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Give it a replaced element
  pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create a Submodel
  submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("moddef2");

  // give it a deletion
  Deletion* del = submod1->createDeletion();
  del->setIdRef("deleteme");
  del->setId("del1");

  // Create a model definition with a parameter to be deleted
  md = compdoc->createModelDefinition();
  md->setId("moddef2");
  param = md->createParameter();
  param->setId("deleteme");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid73)
{
  ConversionProperties* props = new ConversionProperties();
  props->addOption("flatten comp");
  props->addOption("leavePorts", false);
  props->addOption("performValidation", false);
  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  SBMLNamespaces sbmlns(3,1,"comp",1);
  int rv;

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = 
           static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the Model
  Model* model=document->createModel();
  model->setId("mainmod");
  CompModelPlugin* mplugin = 
                   static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  
  // create a Submodel
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("moddef1");

  //Delete the deletion (OK, but can't also reference it, below)
  Deletion* del = submod1->createDeletion();
  del->setIdRef("del1");

  // create a Model Definition
  ModelDefinition* md=compdoc->createModelDefinition();
  md->setId("moddef1");
  mplugin = static_cast<CompModelPlugin*>(md->getPlugin("comp"));
  
  // create a Parameter in it.
  Parameter* param = md->createParameter();
  param->setId("p1");
  param->setConstant(true);

  //Give it a replaced element
  CompSBasePlugin* pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  ReplacedElement* re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create another Parameter in it.
  param = md->createParameter();
  param->setId("p2");
  param->setConstant(true);

  //Give it a replaced element
  pcomp = static_cast<CompSBasePlugin*>(param->getPlugin("comp"));
  re = pcomp->createReplacedElement();
  re->setSubmodelRef("submod1");
  re->setDeletion("del1");
  
  // create a Submodel
  submod1 = mplugin->createSubmodel();
  submod1->setId("submod1");
  submod1->setModelRef("moddef2");

  // give it a deletion
  del = submod1->createDeletion();
  del->setIdRef("deleteme");
  del->setId("del1");

  // Create a model definition with a parameter to be deleted
  md = compdoc->createModelDefinition();
  md->setId("moddef2");
  param = md->createParameter();
  param->setId("deleteme");
  param->setConstant(true);

  //Now try to flatten it
  converter->setDocument(document);
  rv = converter->convert();
  fail_unless(rv==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = document->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete document;
  delete converter;
}
END_TEST


START_TEST(test_comp_flatten_invalid74)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "multi_replaced_by.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();
  fail_unless(result==LIBSBML_OPERATION_FAILED);
  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 2);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompNoMultipleReferences) == true);
  
  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid75)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", true);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "del_only_specref.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getNumErrors() == 0);
  fail_unless(doc->getModel() != NULL);

  converter->setDocument(doc);
  int result = converter->convert();

  fail_unless( result == LIBSBML_CONV_INVALID_SRC_DOCUMENT);

  SBMLErrorLog* errors = doc->getErrorLog();

  fail_unless(errors->getNumErrors() == 3);
  fail_unless(errors->contains(CompLineNumbersUnreliable) == true);
  fail_unless(errors->contains(CompFlatModelNotValid) == true);
  fail_unless(errors->contains(NoReactantsOrProducts) == true);

  delete doc;
  delete converter;
}
END_TEST

START_TEST(test_comp_flatten_invalid77)
{
  ConversionProperties* props = new ConversionProperties();
  
  props->addOption("flatten comp");
  props->addOption("performValidation", false);

  SBMLConverter* converter = 
    SBMLConverterRegistry::getInstance().getConverterFor(*props);
  
  // load document
  string dir(TestDataDirectory);
  string fileName = dir + "replacedby_unknown.xml";  
  SBMLDocument* doc = readSBMLFromFile(fileName.c_str());
  string origdoc = writeSBMLToString(doc);

  // fail if there is no model 
  //(readSBMLFromFile always returns a valid document)
  fail_unless(doc->getModel() != NULL);

  SBMLErrorLog* errors = doc->getErrorLog();
  fail_unless(doc->getNumErrors() == 1);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);

  converter->setDocument(doc);
  int result = converter->convert();
  string newdoc = writeSBMLToString(doc);

  fail_unless( result == LIBSBML_OPERATION_FAILED);
  fail_unless(newdoc == origdoc); //Failing to convert should always leave the original document intact.

  fail_unless(errors->getNumErrors() == 4);
  fail_unless(errors->contains(UnrequiredPackagePresent) == true);
  fail_unless(errors->contains(CompFlatteningNotRecognisedNotReqd) == true);
  fail_unless(errors->contains(CompModelFlatteningFailed) == true);
  fail_unless(errors->contains(CompIdRefMayReferenceUnknownPackage) == true);

  delete doc;
  delete converter;
}
END_TEST


Suite *
create_suite_TestFlatteningErrorMessages (void)
{ 
  TCase *tcase = tcase_create("SBMLCompFlatteningErrorMessages");
  Suite *suite = suite_create("SBMLCompFlatteningErrorMessages");
  
  tcase_add_test(tcase, test_comp_flatten_invalid);
  tcase_add_test(tcase, test_comp_flatten_invalid2);
  tcase_add_test(tcase, test_comp_flatten_invalid3);
  tcase_add_test(tcase, test_comp_flatten_invalid4);
  tcase_add_test(tcase, test_comp_flatten_invalid5);
  tcase_add_test(tcase, test_comp_flatten_invalid6);
  tcase_add_test(tcase, test_comp_flatten_invalid7);
  tcase_add_test(tcase, test_comp_flatten_invalid8);
  tcase_add_test(tcase, test_comp_flatten_invalid9);
  tcase_add_test(tcase, test_comp_flatten_invalid10);
  tcase_add_test(tcase, test_comp_flatten_invalid11);
  tcase_add_test(tcase, test_comp_flatten_invalid12);
  tcase_add_test(tcase, test_comp_flatten_invalid13);
  tcase_add_test(tcase, test_comp_flatten_invalid14);
  tcase_add_test(tcase, test_comp_flatten_invalid15);
  tcase_add_test(tcase, test_comp_flatten_invalid16);
  tcase_add_test(tcase, test_comp_flatten_invalid17);
  tcase_add_test(tcase, test_comp_flatten_invalid18);
  tcase_add_test(tcase, test_comp_flatten_invalid19);
  tcase_add_test(tcase, test_comp_flatten_invalid20);
  tcase_add_test(tcase, test_comp_flatten_invalid21);
  tcase_add_test(tcase, test_comp_flatten_invalid22);
  tcase_add_test(tcase, test_comp_flatten_invalid23);
  tcase_add_test(tcase, test_comp_flatten_invalid24);
  tcase_add_test(tcase, test_comp_flatten_invalid25);
  tcase_add_test(tcase, test_comp_flatten_invalid26);
  tcase_add_test(tcase, test_comp_flatten_invalid27);
  tcase_add_test(tcase, test_comp_flatten_invalid28);
  tcase_add_test(tcase, test_comp_flatten_invalid29);
  tcase_add_test(tcase, test_comp_flatten_invalid30);
  tcase_add_test(tcase, test_comp_flatten_invalid31);
  tcase_add_test(tcase, test_comp_flatten_invalid32);
  tcase_add_test(tcase, test_comp_flatten_invalid33);
  tcase_add_test(tcase, test_comp_flatten_invalid34);
  tcase_add_test(tcase, test_comp_flatten_invalid35);
  tcase_add_test(tcase, test_comp_flatten_invalid36);
  tcase_add_test(tcase, test_comp_flatten_invalid37);
  tcase_add_test(tcase, test_comp_flatten_invalid38);
  tcase_add_test(tcase, test_comp_flatten_invalid39);
  tcase_add_test(tcase, test_comp_flatten_invalid40);
  tcase_add_test(tcase, test_comp_flatten_invalid41);
  tcase_add_test(tcase, test_comp_flatten_invalid42);
  tcase_add_test(tcase, test_comp_flatten_invalid43);
  tcase_add_test(tcase, test_comp_flatten_invalid44);
  tcase_add_test(tcase, test_comp_flatten_invalid45);
  tcase_add_test(tcase, test_comp_flatten_invalid46);
  tcase_add_test(tcase, test_comp_flatten_invalid47);
  tcase_add_test(tcase, test_comp_flatten_invalid48);
  tcase_add_test(tcase, test_comp_flatten_invalid49);
  tcase_add_test(tcase, test_comp_flatten_invalid50);
  tcase_add_test(tcase, test_comp_flatten_invalid51);
  tcase_add_test(tcase, test_comp_flatten_invalid52);
  tcase_add_test(tcase, test_comp_flatten_invalid53);
  tcase_add_test(tcase, test_comp_flatten_invalid54);
  tcase_add_test(tcase, test_comp_flatten_invalid55);
  tcase_add_test(tcase, test_comp_flatten_invalid56);
  tcase_add_test(tcase, test_comp_flatten_invalid57);
  tcase_add_test(tcase, test_comp_flatten_invalid58);
  tcase_add_test(tcase, test_comp_flatten_invalid59);
  tcase_add_test(tcase, test_comp_flatten_invalid60);
  tcase_add_test(tcase, test_comp_flatten_invalid61);
  tcase_add_test(tcase, test_comp_flatten_invalid62);
  tcase_add_test(tcase, test_comp_flatten_invalid63);
  tcase_add_test(tcase, test_comp_flatten_invalid64);
  tcase_add_test(tcase, test_comp_flatten_invalid65);
  tcase_add_test(tcase, test_comp_flatten_invalid66);
  tcase_add_test(tcase, test_comp_flatten_invalid67);
  tcase_add_test(tcase, test_comp_flatten_invalid68);
  tcase_add_test(tcase, test_comp_flatten_invalid69);
  tcase_add_test(tcase, test_comp_flatten_invalid70);
  tcase_add_test(tcase, test_comp_flatten_invalid71);
  tcase_add_test(tcase, test_comp_flatten_invalid72);
  tcase_add_test(tcase, test_comp_flatten_invalid73);
  tcase_add_test(tcase, test_comp_flatten_invalid74);
  tcase_add_test(tcase, test_comp_flatten_invalid75);
//  tcase_add_test(tcase, test_comp_flatten_invalid76);
  tcase_add_test(tcase, test_comp_flatten_invalid77);

  tcase_add_test(tcase, test_comp_flatten_invalid_core);
  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


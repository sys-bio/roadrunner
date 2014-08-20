/**
 * \file    TestSBaseObjects.cpp
 * \brief   Implementation of the Tests for the Comp sbase objects
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

#include <sbml/SBMLReader.h>
#include <sbml/SBMLTypes.h>

#include <string>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;

START_TEST (test_comp_deletion)
{
  Deletion del(3, 1);
  fail_unless(del.getElementName()=="deletion");
  fail_unless(del.getTypeCode()==SBML_COMP_DELETION);

  fail_unless(del.isSetId()==false);
  fail_unless(del.setId("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(del.setId("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getId()=="ID1");
  fail_unless(del.isSetId()==true);
  fail_unless(del.unsetId()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetId()==false);

  fail_unless(del.isSetName()==false);
  fail_unless(del.setName("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.setName("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getName()=="ID1");
  fail_unless(del.isSetName()==true);
  fail_unless(del.unsetName()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetName()==false);

  fail_unless(del.isSetMetaIdRef()==false);
  fail_unless(del.setMetaIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(del.setMetaIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getMetaIdRef()=="ID1");
  fail_unless(del.isSetMetaIdRef()==true);
  fail_unless(del.unsetMetaIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetMetaIdRef()==false);

  fail_unless(del.isSetPortRef()==false);
  fail_unless(del.setPortRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(del.setPortRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getPortRef()=="ID1");
  fail_unless(del.isSetPortRef()==true);
  fail_unless(del.unsetPortRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetPortRef()==false);

  fail_unless(del.isSetIdRef()==false);
  fail_unless(del.setIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(del.setIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getIdRef()=="ID1");
  fail_unless(del.isSetIdRef()==true);
  fail_unless(del.unsetIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetIdRef()==false);

  fail_unless(del.isSetUnitRef()==false);
  fail_unless(del.setUnitRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(del.setUnitRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.getUnitRef()=="ID1");
  fail_unless(del.isSetUnitRef()==true);
  fail_unless(del.unsetUnitRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetUnitRef()==false);

  fail_unless(del.isSetSBaseRef()==false);
  fail_unless(del.createSBaseRef()!=NULL);
  fail_unless(del.getSBaseRef()!=NULL);
  fail_unless(del.isSetSBaseRef()==true);
  fail_unless(del.unsetSBaseRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(del.isSetSBaseRef()==false);

  fail_unless(del.getNumReferents()==0);
  del.setIdRef("ID1");
  fail_unless(del.getNumReferents()==1);
  fail_unless(del.setUnitRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(del.setPortRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(del.setMetaIdRef("ID1") == LIBSBML_OPERATION_FAILED);
}
END_TEST

START_TEST (test_comp_externalmodeldefinition)
{
  ExternalModelDefinition emd(3, 1);
  fail_unless(emd.getElementName()=="externalModelDefinition");
  fail_unless(emd.getTypeCode()==SBML_COMP_EXTERNALMODELDEFINITION);

  fail_unless(emd.isSetId()==false);
  fail_unless(emd.setId("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(emd.setId("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.getId()=="ID1");
  fail_unless(emd.isSetId()==true);
  fail_unless(emd.unsetId()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.isSetId()==false);

  fail_unless(emd.isSetName()==false);
  fail_unless(emd.setName("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.setName("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.getName()=="ID1");
  fail_unless(emd.isSetName()==true);
  fail_unless(emd.unsetName()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.isSetName()==false);

  fail_unless(emd.isSetModelRef()==false);
  fail_unless(emd.setModelRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(emd.setModelRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.getModelRef()=="ID1");
  fail_unless(emd.isSetModelRef()==true);
  fail_unless(emd.unsetModelRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.isSetModelRef()==false);

  fail_unless(emd.isSetMd5()==false);
  fail_unless(emd.setMd5("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.setMd5("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.getMd5()=="ID1");
  fail_unless(emd.isSetMd5()==true);
  fail_unless(emd.unsetMd5()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.isSetMd5()==false);

  fail_unless(emd.isSetSource()==false);
  fail_unless(emd.setSource("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.setSource("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.getSource()=="ID1");
  fail_unless(emd.isSetSource()==true);
  fail_unless(emd.unsetSource()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(emd.isSetSource()==false);

}
END_TEST

START_TEST (test_comp_modeldefinition)
{
  ModelDefinition emd(3, 1);
  fail_unless(emd.getElementName()=="modelDefinition");
  fail_unless(emd.getTypeCode()==SBML_COMP_MODELDEFINITION);
}
END_TEST

  
START_TEST (test_comp_port)
{
  Port port(3, 1);
  fail_unless(port.getElementName()=="port");
  fail_unless(port.getTypeCode()==SBML_COMP_PORT);

  fail_unless(port.isSetId()==false);
  fail_unless(port.setId("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(port.setId("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.getId()=="ID1");
  fail_unless(port.isSetId()==true);
  fail_unless(port.unsetId()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetId()==false);

  fail_unless(port.isSetName()==false);
  fail_unless(port.setName("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.setName("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.getName()=="ID1");
  fail_unless(port.isSetName()==true);
  fail_unless(port.unsetName()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetName()==false);

  fail_unless(port.isSetMetaIdRef()==false);
  fail_unless(port.setMetaIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(port.setMetaIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.getMetaIdRef()=="ID1");
  fail_unless(port.isSetMetaIdRef()==true);
  fail_unless(port.unsetMetaIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetMetaIdRef()==false);

  fail_unless(port.isSetPortRef()==false);
  fail_unless(port.setPortRef("1&d")==LIBSBML_OPERATION_FAILED);
  fail_unless(port.setPortRef("ID1")==LIBSBML_OPERATION_FAILED);
  fail_unless(port.unsetPortRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetPortRef()==false);

  fail_unless(port.isSetIdRef()==false);
  fail_unless(port.setIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(port.setIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.getIdRef()=="ID1");
  fail_unless(port.isSetIdRef()==true);
  fail_unless(port.unsetIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetIdRef()==false);

  fail_unless(port.isSetUnitRef()==false);
  fail_unless(port.setUnitRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(port.setUnitRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.getUnitRef()=="ID1");
  fail_unless(port.isSetUnitRef()==true);
  fail_unless(port.unsetUnitRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetUnitRef()==false);

  fail_unless(port.isSetSBaseRef()==false);
  fail_unless(port.createSBaseRef()!=NULL);
  fail_unless(port.getSBaseRef()!=NULL);
  fail_unless(port.isSetSBaseRef()==true);
  fail_unless(port.unsetSBaseRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(port.isSetSBaseRef()==false);

  fail_unless(port.getNumReferents()==0);
  port.setIdRef("ID1");
  fail_unless(port.getNumReferents()==1);
  fail_unless(port.setUnitRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(port.setPortRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(port.setMetaIdRef("ID1") == LIBSBML_OPERATION_FAILED);
}
END_TEST


START_TEST (test_comp_replacedby)
{
  ReplacedBy rb(3, 1);
  fail_unless(rb.getElementName()=="replacedBy");
  fail_unless(rb.getTypeCode()==SBML_COMP_REPLACEDBY);

  fail_unless(rb.isSetMetaIdRef()==false);
  fail_unless(rb.setMetaIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(rb.setMetaIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.getMetaIdRef()=="ID1");
  fail_unless(rb.isSetMetaIdRef()==true);
  fail_unless(rb.unsetMetaIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetMetaIdRef()==false);

  fail_unless(rb.isSetPortRef()==false);
  fail_unless(rb.setPortRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(rb.setPortRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.getPortRef()=="ID1");
  fail_unless(rb.isSetPortRef()==true);
  fail_unless(rb.unsetPortRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetPortRef()==false);

  fail_unless(rb.isSetIdRef()==false);
  fail_unless(rb.setIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(rb.setIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.getIdRef()=="ID1");
  fail_unless(rb.isSetIdRef()==true);
  fail_unless(rb.unsetIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetIdRef()==false);

  fail_unless(rb.isSetUnitRef()==false);
  fail_unless(rb.setUnitRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(rb.setUnitRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.getUnitRef()=="ID1");
  fail_unless(rb.isSetUnitRef()==true);
  fail_unless(rb.unsetUnitRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetUnitRef()==false);

  fail_unless(rb.isSetSBaseRef()==false);
  fail_unless(rb.createSBaseRef()!=NULL);
  fail_unless(rb.getSBaseRef()!=NULL);
  fail_unless(rb.isSetSBaseRef()==true);
  fail_unless(rb.unsetSBaseRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetSBaseRef()==false);

  fail_unless(rb.isSetSubmodelRef()==false);
  fail_unless(rb.setSubmodelRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(rb.setSubmodelRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.getSubmodelRef()=="ID1");
  fail_unless(rb.isSetSubmodelRef()==true);
  fail_unless(rb.unsetSubmodelRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(rb.isSetSubmodelRef()==false);

  fail_unless(rb.getNumReferents()==0);
  rb.setIdRef("ID1");
  fail_unless(rb.getNumReferents()==1);
  fail_unless(rb.setUnitRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(rb.setPortRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(rb.setMetaIdRef("ID1") == LIBSBML_OPERATION_FAILED);
}
END_TEST

START_TEST (test_comp_replacedelement)
{
  ReplacedElement re(3, 1);
  fail_unless(re.getElementName()=="replacedElement");
  fail_unless(re.getTypeCode()==SBML_COMP_REPLACEDELEMENT);

  fail_unless(re.isSetMetaIdRef()==false);
  fail_unless(re.setMetaIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setMetaIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getMetaIdRef()=="ID1");
  fail_unless(re.isSetMetaIdRef()==true);
  fail_unless(re.unsetMetaIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetMetaIdRef()==false);

  fail_unless(re.isSetPortRef()==false);
  fail_unless(re.setPortRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setPortRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getPortRef()=="ID1");
  fail_unless(re.isSetPortRef()==true);
  fail_unless(re.unsetPortRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetPortRef()==false);

  fail_unless(re.isSetIdRef()==false);
  fail_unless(re.setIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getIdRef()=="ID1");
  fail_unless(re.isSetIdRef()==true);
  fail_unless(re.unsetIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetIdRef()==false);

  fail_unless(re.isSetUnitRef()==false);
  fail_unless(re.setUnitRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setUnitRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getUnitRef()=="ID1");
  fail_unless(re.isSetUnitRef()==true);
  fail_unless(re.unsetUnitRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetUnitRef()==false);

  fail_unless(re.isSetDeletion()==false);
  fail_unless(re.setDeletion("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setDeletion("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getDeletion()=="ID1");
  fail_unless(re.isSetDeletion()==true);
  fail_unless(re.unsetDeletion()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetDeletion()==false);

  fail_unless(re.isSetSBaseRef()==false);
  fail_unless(re.createSBaseRef()!=NULL);
  fail_unless(re.getSBaseRef()!=NULL);
  fail_unless(re.isSetSBaseRef()==true);
  fail_unless(re.unsetSBaseRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetSBaseRef()==false);

  fail_unless(re.isSetSubmodelRef()==false);
  fail_unless(re.setSubmodelRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setSubmodelRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getSubmodelRef()=="ID1");
  fail_unless(re.isSetSubmodelRef()==true);
  fail_unless(re.unsetSubmodelRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetSubmodelRef()==false);

  fail_unless(re.isSetConversionFactor()==false);
  fail_unless(re.setConversionFactor("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(re.setConversionFactor("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.getConversionFactor()=="ID1");
  fail_unless(re.isSetConversionFactor()==true);
  fail_unless(re.unsetConversionFactor()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(re.isSetConversionFactor()==false);

  fail_unless(re.getNumReferents()==0);
  re.setIdRef("ID1");
  fail_unless(re.getNumReferents()==1);
  fail_unless(re.setUnitRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(re.setDeletion("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(re.setPortRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(re.setMetaIdRef("ID1") == LIBSBML_OPERATION_FAILED);
}
END_TEST


START_TEST (test_comp_sbaseref)
{
  SBaseRef sbr(3, 1);
  fail_unless(sbr.getElementName()=="sBaseRef");
  fail_unless(sbr.getTypeCode()==SBML_COMP_SBASEREF);

  fail_unless(sbr.isSetMetaIdRef()==false);
  fail_unless(sbr.setMetaIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(sbr.setMetaIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.getMetaIdRef()=="ID1");
  fail_unless(sbr.isSetMetaIdRef()==true);
  fail_unless(sbr.unsetMetaIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.isSetMetaIdRef()==false);

  fail_unless(sbr.isSetPortRef()==false);
  fail_unless(sbr.setPortRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(sbr.setPortRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.getPortRef()=="ID1");
  fail_unless(sbr.isSetPortRef()==true);
  fail_unless(sbr.unsetPortRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.isSetPortRef()==false);

  fail_unless(sbr.isSetIdRef()==false);
  fail_unless(sbr.setIdRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(sbr.setIdRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.getIdRef()=="ID1");
  fail_unless(sbr.isSetIdRef()==true);
  fail_unless(sbr.unsetIdRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.isSetIdRef()==false);

  fail_unless(sbr.isSetUnitRef()==false);
  fail_unless(sbr.setUnitRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(sbr.setUnitRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.getUnitRef()=="ID1");
  fail_unless(sbr.isSetUnitRef()==true);
  fail_unless(sbr.unsetUnitRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.isSetUnitRef()==false);

  fail_unless(sbr.isSetSBaseRef()==false);
  fail_unless(sbr.createSBaseRef()!=NULL);
  fail_unless(sbr.getSBaseRef()!=NULL);
  fail_unless(sbr.isSetSBaseRef()==true);
  fail_unless(sbr.unsetSBaseRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(sbr.isSetSBaseRef()==false);

  fail_unless(sbr.getNumReferents()==0);
  sbr.setIdRef("ID1");
  fail_unless(sbr.getNumReferents()==1);
  fail_unless(sbr.setUnitRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(sbr.setPortRef("ID1") == LIBSBML_OPERATION_FAILED);
  fail_unless(sbr.setMetaIdRef("ID1") == LIBSBML_OPERATION_FAILED);
}
END_TEST

START_TEST (test_comp_sbaseref_copy)
{
  SBaseRef sbr(3, 1);
  sbr.setIdRef("sub1");
  SBaseRef* sbr2 = sbr.createSBaseRef();
  sbr2->setIdRef("sub2");
  SBaseRef* sbr3 = sbr2->createSBaseRef();
  sbr3->setIdRef("comp1");

  SBaseRef sbrcopy = sbr;
  sbr3->setIdRef("S1");
  SBaseRef sbrcopy2(sbr);
  sbr3->setIdRef("J0");

  SBaseRef* sbrcopy_sub3 = sbrcopy.getSBaseRef()->getSBaseRef();
  fail_unless(sbrcopy_sub3 != NULL);
  fail_unless(sbrcopy_sub3->getIdRef() == "comp1");

  SBaseRef* sbrcopy2_sub3 = sbrcopy2.getSBaseRef()->getSBaseRef();
  fail_unless(sbrcopy2_sub3 != NULL);
  fail_unless(sbrcopy2_sub3->getIdRef() == "S1");

  fail_unless(sbr3->getIdRef() == "J0");

}
END_TEST

START_TEST (test_comp_submodel)
{
  Submodel submod(3, 1);
  fail_unless(submod.getElementName()=="submodel");
  fail_unless(submod.getTypeCode()==SBML_COMP_SUBMODEL);

  fail_unless(submod.isSetId()==false);
  fail_unless(submod.setId("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(submod.setId("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.getId()=="ID1");
  fail_unless(submod.isSetId()==true);
  fail_unless(submod.unsetId()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetId()==false);

  fail_unless(submod.isSetName()==false);
  fail_unless(submod.setName("1&d")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.setName("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.getName()=="ID1");
  fail_unless(submod.isSetName()==true);
  fail_unless(submod.unsetName()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetName()==false);

  fail_unless(submod.isSetModelRef()==false);
  fail_unless(submod.setModelRef("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(submod.setModelRef("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.getModelRef()=="ID1");
  fail_unless(submod.isSetModelRef()==true);
  fail_unless(submod.unsetModelRef()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetModelRef()==false);

  fail_unless(submod.isSetSubstanceConversionFactor()==false);
  fail_unless(submod.setSubstanceConversionFactor("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  // since there is no such attribute, the test below will fail. might as well take them out
  //fail_unless(submod.setSubstanceConversionFactor("ID1")==LIBSBML_OPERATION_SUCCESS);
  //fail_unless(submod.getSubstanceConversionFactor()=="ID1");
  //fail_unless(submod.isSetSubstanceConversionFactor()==true);
  //fail_unless(submod.unsetSubstanceConversionFactor()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetSubstanceConversionFactor()==false);

  fail_unless(submod.isSetTimeConversionFactor()==false);
  fail_unless(submod.setTimeConversionFactor("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(submod.setTimeConversionFactor("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.getTimeConversionFactor()=="ID1");
  fail_unless(submod.isSetTimeConversionFactor()==true);
  fail_unless(submod.unsetTimeConversionFactor()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetTimeConversionFactor()==false);

  fail_unless(submod.isSetExtentConversionFactor()==false);
  fail_unless(submod.setExtentConversionFactor("1&d")==LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(submod.setExtentConversionFactor("ID1")==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.getExtentConversionFactor()=="ID1");
  fail_unless(submod.isSetExtentConversionFactor()==true);
  fail_unless(submod.unsetExtentConversionFactor()==LIBSBML_OPERATION_SUCCESS);
  fail_unless(submod.isSetExtentConversionFactor()==false);

  fail_unless(submod.getNumDeletions()==0);
  Deletion* del = submod.createDeletion();
  del->setId("del1");
  fail_unless(submod.getNumDeletions()==1);
  fail_unless(submod.addDeletion(NULL)==LIBSBML_INVALID_OBJECT);
  Deletion del2(3,1);
  del2.setId("del2");
  fail_unless(submod.addDeletion(&del2)==LIBSBML_INVALID_OBJECT);
  del2.setIdRef("ID1");
  fail_unless(submod.addDeletion(&del2)==LIBSBML_OPERATION_SUCCESS);
  Deletion* delref = submod.getDeletion("del2");
  fail_unless(delref->getId()=="del2");
  delref = submod.getDeletion(0);
  fail_unless(delref->getId()=="del1");
  del->setIdRef("ID1");
  fail_unless(delref->getIdRef()=="ID1");
  fail_unless(submod.removeDeletion(3)==NULL);
  fail_unless(submod.removeDeletion(0)==del);
  fail_unless(submod.getDeletion("del1")==NULL);

}
END_TEST

START_TEST (test_comp_listofdeletions)
{
  ListOfDeletions list(3, 1);
  fail_unless(list.getElementName()=="listOfDeletions");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_DELETION);
}
END_TEST

START_TEST (test_comp_listofexternalmodeldefinitions)
{
  ListOfExternalModelDefinitions list(3, 1);
  fail_unless(list.getElementName()=="listOfExternalModelDefinitions");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_EXTERNALMODELDEFINITION);
}
END_TEST

START_TEST (test_comp_listofmodeldefinitions)
{
  ListOfModelDefinitions list(3, 1);
  fail_unless(list.getElementName()=="listOfModelDefinitions");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_MODELDEFINITION);
}
END_TEST

START_TEST (test_comp_listofports)
{
  ListOfPorts list(3, 1);
  fail_unless(list.getElementName()=="listOfPorts");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_PORT);
}
END_TEST

START_TEST (test_comp_listofreplacedelements)
{
  ListOfReplacedElements list(3, 1);
  fail_unless(list.getElementName()=="listOfReplacedElements");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_REPLACEDELEMENT);
}
END_TEST

START_TEST (test_comp_listofsubmodels)
{
  ListOfSubmodels list(3, 1);
  fail_unless(list.getElementName()=="listOfSubmodels");
  fail_unless(list.getTypeCode()==SBML_LIST_OF);
  fail_unless(list.getItemTypeCode()==SBML_COMP_SUBMODEL);
}
END_TEST

Suite *
create_suite_TestCompSBaseObjects(void)
{ 
  TCase *tcase = tcase_create("SBMLCompSBaseObjects");
  Suite *suite = suite_create("SBMLCompSBaseObjects");
  
  tcase_add_test(tcase, test_comp_deletion);
  tcase_add_test(tcase, test_comp_externalmodeldefinition);
  tcase_add_test(tcase, test_comp_modeldefinition);
  tcase_add_test(tcase, test_comp_port);
  tcase_add_test(tcase, test_comp_replacedby);
  tcase_add_test(tcase, test_comp_replacedelement);
  tcase_add_test(tcase, test_comp_sbaseref);
  tcase_add_test(tcase, test_comp_sbaseref_copy);
  tcase_add_test(tcase, test_comp_submodel);
  tcase_add_test(tcase, test_comp_listofdeletions);
  tcase_add_test(tcase, test_comp_listofexternalmodeldefinitions);
  tcase_add_test(tcase, test_comp_listofmodeldefinitions);
  tcase_add_test(tcase, test_comp_listofports);
  tcase_add_test(tcase, test_comp_listofreplacedelements);
  tcase_add_test(tcase, test_comp_listofsubmodels);

  tcase_add_test(tcase, test_comp_listofsubmodels);

  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS


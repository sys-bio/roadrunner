/**
 * \file    TestSubmodel.c
 * \brief   Submodel unit tests
 * \author  Sarah Keating
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

#include <sbml/packages/comp/sbml/Submodel.h>
#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/sbml/SBaseRef.h>
#include <sbml/packages/comp/sbml/Deletion.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static Submodel_t *P;


void
SubmodelTest_setup (void)
{
  P = Submodel_create(3, 1, 1);

  if (P == NULL)
  {
    fail("Submodel_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
SubmodelTest_teardown (void)
{
  Submodel_free(P);
}


START_TEST (test_comp_submodel_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_SUBMODEL );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( Submodel_getId     (P) == NULL );
  fail_unless( Submodel_getName   (P) == NULL );
  fail_unless( Submodel_getModelRef   (P) == NULL );
  fail_unless( Submodel_getSubstanceConversionFactor   (P) == NULL );
  fail_unless( Submodel_getTimeConversionFactor   (P) == NULL );
  fail_unless( Submodel_getExtentConversionFactor   (P) == NULL );
  fail_unless( Submodel_getDeletion   (P, 0) == NULL );

  fail_unless( !Submodel_isSetId     (P) );
  fail_unless( !Submodel_isSetName   (P) );
  fail_unless( !Submodel_isSetModelRef   (P) );
  fail_unless( !Submodel_isSetSubstanceConversionFactor   (P) );
  fail_unless( !Submodel_isSetTimeConversionFactor   (P) );
  fail_unless( !Submodel_isSetExtentConversionFactor   (P) );
  fail_unless( Submodel_getNumDeletions(P)==0 );
}
END_TEST


START_TEST (test_comp_submodel_free_NULL)
{
  Submodel_free(NULL);
}
END_TEST


START_TEST (test_comp_submodel_id)
{
  const char *id = "mitochondria";


  fail_unless( !Submodel_isSetId(P) );
  
  Submodel_setId(P, id);

  fail_unless( !strcmp(Submodel_getId(P), id) );
  fail_unless( Submodel_isSetId(P) );

  if (Submodel_getId(P) == id)
  {
    fail("Submodel_setId(...) did not make a copy of string.");
  }
 
  Submodel_unsetId(P);
  
  fail_unless( !Submodel_isSetId(P) );

  if (Submodel_getId(P) != NULL)
  {
    fail("Submodel_unsetId(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_submodel_name)
{
  const char *name = "My_Favorite_Factory";


  fail_unless( !Submodel_isSetName(P) );

  Submodel_setName(P, name);

  fail_unless( !strcmp(Submodel_getName(P), name) );
  fail_unless( Submodel_isSetName(P) );

  if (Submodel_getName(P) == name)
  {
    fail("Submodel_setName(...) did not make a copy of string.");
  }

  Submodel_unsetName(P);
  
  fail_unless( !Submodel_isSetName(P) );

  if (Submodel_getName(P) != NULL)
  {
    fail("Submodel_unsetName(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_submodel_modelRef)
{
  const char *modelRef = "My_Favorite_Factory";


  fail_unless( !Submodel_isSetModelRef(P) );

  Submodel_setModelRef(P, modelRef);

  fail_unless( !strcmp(Submodel_getModelRef(P), modelRef) );
  fail_unless( Submodel_isSetModelRef(P) );

  if (Submodel_getModelRef(P) == modelRef)
  {
    fail("Submodel_setModelRef(...) did not make a copy of string.");
  }

  Submodel_unsetModelRef(P);
  
  fail_unless( !Submodel_isSetModelRef(P) );

  if (Submodel_getModelRef(P) != NULL)
  {
    fail("Submodel_unsetModelRef(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_submodel_substanceConversionFactor)
{
  //There is no substance conversion factor any more--all related functions should fail.
  const char *substanceConversionFactor = "My_Favorite_Factory";

  fail_unless( !Submodel_isSetSubstanceConversionFactor(P) );

  fail_unless(Submodel_setSubstanceConversionFactor(P, substanceConversionFactor) == LIBSBML_INVALID_ATTRIBUTE_VALUE) ;

  fail_unless( Submodel_getSubstanceConversionFactor(P) == NULL );
  fail_unless( !Submodel_isSetSubstanceConversionFactor(P) );

  fail_unless(Submodel_unsetSubstanceConversionFactor(P) == LIBSBML_OPERATION_FAILED);
}
END_TEST


START_TEST (test_comp_submodel_timeConversionFactor)
{
  const char *timeConversionFactor = "My_Favorite_Factory";


  fail_unless( !Submodel_isSetTimeConversionFactor(P) );

  Submodel_setTimeConversionFactor(P, timeConversionFactor);

  fail_unless( !strcmp(Submodel_getTimeConversionFactor(P), timeConversionFactor) );
  fail_unless( Submodel_isSetTimeConversionFactor(P) );

  if (Submodel_getTimeConversionFactor(P) == timeConversionFactor)
  {
    fail("Submodel_setTimeConversionFactor(...) did not make a copy of string.");
  }

  Submodel_unsetTimeConversionFactor(P);
  
  fail_unless( !Submodel_isSetTimeConversionFactor(P) );

  if (Submodel_getTimeConversionFactor(P) != NULL)
  {
    fail("Submodel_unsetTimeConversionFactor(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_submodel_extentConversionFactor)
{
  const char *extentConversionFactor = "My_Favorite_Factory";


  fail_unless( !Submodel_isSetExtentConversionFactor(P) );

  Submodel_setExtentConversionFactor(P, extentConversionFactor);

  fail_unless( !strcmp(Submodel_getExtentConversionFactor(P), extentConversionFactor) );
  fail_unless( Submodel_isSetExtentConversionFactor(P) );

  if (Submodel_getExtentConversionFactor(P) == extentConversionFactor)
  {
    fail("Submodel_setExtentConversionFactor(...) did not make a copy of string.");
  }

  Submodel_unsetExtentConversionFactor(P);
  
  fail_unless( !Submodel_isSetExtentConversionFactor(P) );

  if (Submodel_getExtentConversionFactor(P) != NULL)
  {
    fail("Submodel_unsetExtentConversionFactor(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_submodel_deletion)
{
  Deletion_t *deletion = Deletion_create(3, 1, 1);
  const char* delname = "deletion";
  SBaseRef_setIdRef((SBaseRef_t*)deletion, "ref_id");


  fail_unless( Submodel_getNumDeletions(P) == 0 );

  Submodel_addDeletion(P, deletion);

  fail_unless( Submodel_getDeletion(P, 0) != NULL);
  fail_unless( Submodel_getNumDeletions(P)==1 );

  if (Submodel_getDeletion(P, 0) == deletion)
  {
    fail("Submodel_addDeletion(...) did not make a copy of the deletion.");
  }

  Submodel_removeDeletion(P, 0);
  
  fail_unless( Submodel_getDeletion(P, 0) == NULL);
  fail_unless( Submodel_getNumDeletions(P)==0 );

  Deletion_setId(deletion, delname);

  Submodel_addDeletion(P, deletion);
  fail_unless( Submodel_getDeletionById(P, delname) != NULL);

  Submodel_removeDeletionById(P, delname);
  fail_unless( Submodel_getDeletionById(P, delname) == NULL);
  fail_unless( Submodel_getNumDeletions(P)==0 );

  Deletion_free(deletion);
}
END_TEST


START_TEST (test_comp_submodel_hasRequiredAttributes )
{
  Submodel_t *P = Submodel_create (3, 1, 1);

  fail_unless ( !Submodel_hasRequiredAttributes(P));

  Submodel_setId(P, "p");

  fail_unless ( !Submodel_hasRequiredAttributes(P));

  SBaseRef_setIdRef((SBaseRef_t*)(P), "s");
  
  fail_unless ( !Submodel_hasRequiredAttributes(P));

  Submodel_setModelRef(P, "sub1");
  
  fail_unless ( Submodel_hasRequiredAttributes(P));

  Submodel_free(P);
}
END_TEST


Suite *
create_suite_TestComp_Submodel (void)
{
  Suite *suite = suite_create("comp_submodel");
  TCase *tcase = tcase_create("comp_submodel");


  tcase_add_checked_fixture( tcase,
                             SubmodelTest_setup,
                             SubmodelTest_teardown );

  tcase_add_test( tcase, test_comp_submodel_create                   );
  tcase_add_test( tcase, test_comp_submodel_free_NULL                );
  tcase_add_test( tcase, test_comp_submodel_id                       );
  tcase_add_test( tcase, test_comp_submodel_name                     );
  tcase_add_test( tcase, test_comp_submodel_modelRef                 );
  tcase_add_test( tcase, test_comp_submodel_substanceConversionFactor);
  tcase_add_test( tcase, test_comp_submodel_timeConversionFactor     );
  tcase_add_test( tcase, test_comp_submodel_extentConversionFactor   );
  tcase_add_test( tcase, test_comp_submodel_deletion                 );
  tcase_add_test( tcase, test_comp_submodel_hasRequiredAttributes    );

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS


/**
 * \file    TestCopyAndClone.cpp
 * \brief   Copy SBML unit tests
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
 * Copyright (C) 2009-2013 jointly by the following organizations: 
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

#include <sbml/common/common.h>

#include <sbml/SBase.h>
#include <sbml/packages/qual/sbml/Input.h>
#include <sbml/packages/qual/sbml/Output.h>
#include <sbml/packages/qual/sbml/FunctionTerm.h>
#include <sbml/packages/qual/sbml/DefaultTerm.h>
#include <sbml/packages/qual/sbml/Transition.h>
#include <sbml/packages/qual/sbml/QualitativeSpecies.h>

#include <sbml/math/ASTNode.h>

#include <check.h>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond */


CK_CPPSTART

static QualPkgNamespaces* GNS = new QualPkgNamespaces();

START_TEST ( test_Input_copyConstructor )
{
    Input* o1=new Input(GNS);
    o1->setId("c");
    o1->setQualitativeSpecies("c2");
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getQualitativeSpecies() == "c2");

    Input* o2=new Input(*o1);

    fail_unless(o2->getId() == "c");
    fail_unless(o2->getQualitativeSpecies() == "c2");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Input_assignmentOperator )
{
    Input* o1=new Input(GNS);
    o1->setId("c");
    o1->setQualitativeSpecies("c2");
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getQualitativeSpecies() == "c2");
    
    Input* o2 = new Input(GNS);;
    (*o2)=*o1;

    fail_unless(o2->getId() == "c");
    fail_unless(o2->getQualitativeSpecies() == "c2");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Input_clone )
{
    Input* o1=new Input(GNS);
    o1->setId("c");
    o1->setQualitativeSpecies("c2");
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getQualitativeSpecies() == "c2");

    Input* o2=o1->clone();
   
    fail_unless(o2->getId() == "c");
    fail_unless(o2->getQualitativeSpecies() == "c2");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Output_copyConstructor )
{
    Output* o1=new Output(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");

    Output* o2=new Output(*o1);

    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST

START_TEST ( test_Output_assignmentOperator )
{
    Output* o1=new Output(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");
    
    Output* o2 = new Output(GNS);;
    (*o2)=*o1;

    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Output_clone )
{
    Output* o1=new Output(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");

    Output* o2=o1->clone();
   
    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_FunctionTerm_copyConstructor )
{
    FunctionTerm* o1=new FunctionTerm(GNS);
    o1->setMetaId("c");
    
    fail_unless(o1->getMetaId() == "c");

    ASTNode * math = new ASTNode (AST_CONSTANT_PI);

    o1->setMath(math);

    delete math;


    fail_unless(o1->getMath() != NULL);

    FunctionTerm* o2 = new FunctionTerm(*o1);

    fail_unless(o2->getMetaId() == "c");
    fail_unless(o2->getMath() != o1->getMath());

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST

START_TEST ( test_FunctionTerm_assignmentOperator )
{
    FunctionTerm* o1=new FunctionTerm(GNS);
    o1->setMetaId("c");
    
    fail_unless(o1->getMetaId() == "c");
    
    ASTNode * math = new ASTNode (AST_CONSTANT_PI);

    o1->setMath(math);

    delete math;

    fail_unless(o1->getMath() != NULL);

    FunctionTerm* o2 = new FunctionTerm(GNS);;
    (*o2)=*o1;

    fail_unless(o2->getMetaId() == "c");
    fail_unless(o2->getMath() != o1->getMath());

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_FunctionTerm_clone )
{
    FunctionTerm* o1=new FunctionTerm(GNS);
    o1->setMetaId("c");
    
    fail_unless(o1->getMetaId() == "c");

    ASTNode * math = new ASTNode (AST_CONSTANT_PI);

    o1->setMath(math);

    delete math;
    fail_unless(o1->getMath() != NULL);

    FunctionTerm* o2=o1->clone();
   
    fail_unless(o2->getMetaId() == "c");
    fail_unless(o2->getMath() != o1->getMath());

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_DefaultTerm_copyConstructor )
{
    DefaultTerm* o1=new DefaultTerm(GNS);
    o1->setResultLevel(1);

    fail_unless(o1->getResultLevel() == 1);

    DefaultTerm* o2=new DefaultTerm(*o1);

    fail_unless(o2->getResultLevel() == 1);

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST

START_TEST ( test_DefaultTerm_assignmentOperator )
{
    DefaultTerm* o1=new DefaultTerm(GNS);
    o1->setResultLevel(1);

    fail_unless(o1->getResultLevel() == 1);

    DefaultTerm* o2 = new DefaultTerm(GNS);
    (*o2)=*o1;

    fail_unless(o2->getResultLevel() == 1);

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_DefaultTerm_clone )
{
    DefaultTerm* o1=new DefaultTerm(GNS);
    o1->setResultLevel(1);

    fail_unless(o1->getResultLevel() == 1);

    DefaultTerm* o2 = o1->clone();

    fail_unless(o2->getResultLevel() == 1);

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Transition_copyConstructor )
{
    Transition* o1=new Transition(GNS);
    o1->setId("c");
    FunctionTerm* ft=new FunctionTerm(GNS);
    ASTNode * math = new ASTNode (AST_CONSTANT_PI);
    ft->setMath(math);
    ft->setResultLevel(0);
    o1->addFunctionTerm(ft);
    DefaultTerm *dt = new DefaultTerm(GNS);
    dt->setResultLevel(0);
    o1->setDefaultTerm(dt);

    delete ft;
    delete math;
    delete dt;
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getNumFunctionTerms() == 1);
    fail_unless(o1->getFunctionTerm(0) != NULL);
    fail_unless(o1->getDefaultTerm() != NULL);

    Transition* o2=new Transition(*o1);

    fail_unless(o2->getId() == "c");
    fail_unless(o2->getNumFunctionTerms() == 1);
    fail_unless(o2->getFunctionTerm(0) != NULL);
    fail_unless(o2->getFunctionTerm(0) != o1->getFunctionTerm(0));
    fail_unless(o2->getDefaultTerm() != o1->getDefaultTerm());

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST

START_TEST ( test_Transition_assignmentOperator )
{
    Transition* o1=new Transition(GNS);
    o1->setId("c");
    FunctionTerm* ft=new FunctionTerm(GNS);
    ASTNode * math = new ASTNode (AST_CONSTANT_PI);
    ft->setMath(math);
    ft->setResultLevel(0);
    o1->addFunctionTerm(ft);
    DefaultTerm *dt = new DefaultTerm(GNS);
    dt->setResultLevel(0);
    o1->setDefaultTerm(dt);

    delete ft;
    delete math;
    delete dt;
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getNumFunctionTerms() == 1);
    fail_unless(o1->getFunctionTerm(0) != NULL);
    fail_unless(o1->getDefaultTerm() != NULL);

    Transition* o2 = new Transition(GNS);;
    (*o2)=*o1;

    fail_unless(o2->getId() == "c");
    fail_unless(o2->getNumFunctionTerms() == 1);
    fail_unless(o2->getFunctionTerm(0) != NULL);
    fail_unless(o2->getFunctionTerm(0) != o1->getFunctionTerm(0));
    fail_unless(o2->getDefaultTerm() != o1->getDefaultTerm());

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_Transition_clone )
{
    Transition* o1=new Transition(GNS);
    o1->setId("c");
    FunctionTerm* ft=new FunctionTerm(GNS);
    ASTNode * math = new ASTNode (AST_CONSTANT_PI);
    ft->setMath(math);
    ft->setResultLevel(0);
    o1->addFunctionTerm(ft);

    delete ft;
    delete math;
    
    fail_unless(o1->getId() == "c");
    fail_unless(o1->getNumFunctionTerms() == 1);
    fail_unless(o1->getFunctionTerm(0) != NULL);

    Transition* o2=o1->clone();
   
    fail_unless(o2->getId() == "c");
    fail_unless(o2->getNumFunctionTerms() == 1);
    fail_unless(o2->getFunctionTerm(0) != NULL);
    fail_unless(o2->getFunctionTerm(0) != o1->getFunctionTerm(0));

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_QualitativeSpecies_copyConstructor )
{
    QualitativeSpecies* o1=new QualitativeSpecies(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");

    QualitativeSpecies* o2=new QualitativeSpecies(*o1);

    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST

START_TEST ( test_QualitativeSpecies_assignmentOperator )
{
    QualitativeSpecies* o1=new QualitativeSpecies(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");
    
    QualitativeSpecies* o2 = new QualitativeSpecies(GNS);;
    (*o2)=*o1;

    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


START_TEST ( test_QualitativeSpecies_clone )
{
    QualitativeSpecies* o1=new QualitativeSpecies(GNS);
    o1->setId("c");
    
    fail_unless(o1->getId() == "c");

    QualitativeSpecies* o2=o1->clone();
   
    fail_unless(o2->getId() == "c");

    fail_unless(o2->getParentSBMLObject() == o1->getParentSBMLObject());

    delete o2;
    delete o1;
}
END_TEST


Suite *
create_suite_CopyAndClone (void)
{
  Suite *suite = suite_create("CopyAndClone");
  TCase *tcase = tcase_create("CopyAndClone");

  tcase_add_test( tcase, test_Input_copyConstructor );
  tcase_add_test( tcase, test_Input_assignmentOperator );
  tcase_add_test( tcase, test_Input_clone );
  tcase_add_test( tcase, test_Output_copyConstructor );
  tcase_add_test( tcase, test_Output_assignmentOperator );
  tcase_add_test( tcase, test_Output_clone );
  tcase_add_test( tcase, test_FunctionTerm_copyConstructor );
  tcase_add_test( tcase, test_FunctionTerm_assignmentOperator );
  tcase_add_test( tcase, test_FunctionTerm_clone );
  tcase_add_test( tcase, test_DefaultTerm_copyConstructor );
  tcase_add_test( tcase, test_DefaultTerm_assignmentOperator );
  tcase_add_test( tcase, test_DefaultTerm_clone );
  tcase_add_test( tcase, test_Transition_copyConstructor );
  tcase_add_test( tcase, test_Transition_assignmentOperator );
  tcase_add_test( tcase, test_Transition_clone );
  tcase_add_test( tcase, test_QualitativeSpecies_copyConstructor );
  tcase_add_test( tcase, test_QualitativeSpecies_assignmentOperator );
  tcase_add_test( tcase, test_QualitativeSpecies_clone );

  suite_add_tcase(suite, tcase);

  return suite;
}
CK_CPPEND


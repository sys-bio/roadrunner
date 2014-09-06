/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    QualCSymbolMathCheck.cpp
 * @brief   checks &lt;ci&gt; element is the id of a component
 * @author  Sarah Keating
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


#include "QualCSymbolMathCheck.h"

#include <sbml/packages/qual/extension/QualModelPlugin.h>
/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus
/**
 * Creates a new Constraint with the given @p id.
 */
QualCSymbolMathCheck::QualCSymbolMathCheck (unsigned int id, Validator& v) : MathMLBase(id, v)
{
}


/**
 * Destroys this Constraint.
 */
QualCSymbolMathCheck::~QualCSymbolMathCheck ()
{
}


void
QualCSymbolMathCheck::check_ (const Model& m, const Model& object)
{
  unsigned int n, i;

  QualModelPlugin* plug = (QualModelPlugin*)(object.getPlugin("qual"));

  if (plug == NULL) return;
  
  /* create a list of local parameters ids */
  for (n = 0; n < plug->getNumTransitions(); n++)
  {
    Transition * tr = plug->getTransition(n);

    for (i = 0; i < tr->getNumFunctionTerms(); i++)
    {
      if (tr->getFunctionTerm(i)->isSetMath() == true)
      {
        checkMath(m, *(tr->getFunctionTerm(i)->getMath()), *tr);
      }
    }
  }
}

/**
  * Checks the MathML of the ASTnode 
  * is appropriate for the function being performed
  *
  * If an inconsistency is found, an error message is logged.
  */
void
QualCSymbolMathCheck::checkMath (const Model& m, const ASTNode& node, const SBase & sb)
{
  /* should not be here but why not catch it rather than crash*/
  if (&(node) == NULL)
  {
    return;
  }

  ASTNodeType_t type = node.getType();
    
  /* if the node is a &lt;ci&gt; element it will have type AST_NAME
   * check that this name is an appropriate component of the model */
  switch (type) 
  {
    case AST_NAME_TIME:
    case AST_FUNCTION_DELAY:

      logMathConflict(node, sb);
      break;

    default:

      checkChildren(m, node, sb);
      break;

  }
}

  
/**
 * @return the error message to use when logging constraint violations.
 * This method is called by logFailure.
 *
 * Returns a message that the given @p id and its corresponding object are
 * in  conflict with an object previously defined.
 */
const string
QualCSymbolMathCheck::getMessage (const ASTNode& node, const SBase& object)
{

  ostringstream msg;

  //msg << getPreamble();
  char * formula = SBML_formulaToString(&node);
  msg << "\nThe formula '" << formula;
  msg << "' in the math element of the " << getTypename(object);
  msg << " uses a csymbol.";
  safe_free(formula);

  return msg.str();
}

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

#endif /* __cplusplus */

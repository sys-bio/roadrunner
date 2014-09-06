/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    UniqueModelIds.cpp
 * @brief   Ensures the appropriate ids within a Model are unique
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

#include <sbml/Model.h>
#include "UniqueModelIds.h"

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new Constraint with the given constraint id.
 */
UniqueModelIds::UniqueModelIds (unsigned int id, CompValidator& v) :
  UniqueCompIdBase(id, v)
{
}


/*
 * Destroys this Constraint.
 */
UniqueModelIds::~UniqueModelIds ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
UniqueModelIds::doCheck (const Model& m)
{
  unsigned int n, size;

  checkId( m );

  const SBMLDocument * doc = m.getSBMLDocument();
  if (doc == NULL)
  {
    return;
  }

  const CompSBMLDocumentPlugin * plug = 
    static_cast <const CompSBMLDocumentPlugin*>(doc->getPlugin("comp"));
  if (plug == NULL)
  {
    return;
  }

  size = plug->getNumExternalModelDefinitions();
  for (n = 0; n < size; ++n) 
  {
    checkId( *(plug->getExternalModelDefinition(n)));
  }

  size = plug->getNumModelDefinitions();
  for (n = 0; n < size; ++n) 
  {
    checkId( *(plug->getModelDefinition(n)));
  }

  reset();
}
#endif /* __cplusplus */ 


LIBSBML_CPP_NAMESPACE_END

/** @endcond */

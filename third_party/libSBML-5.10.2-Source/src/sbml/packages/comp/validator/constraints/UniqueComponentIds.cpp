/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    UniqueComponentIds.cpp
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
#include "UniqueComponentIds.h"

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new Constraint with the given constraint id.
 */
UniqueComponentIds::UniqueComponentIds (unsigned int id, CompValidator& v) :
  UniqueCompIdBase(id, v)
{
}


/*
 * Destroys this Constraint.
 */
UniqueComponentIds::~UniqueComponentIds ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
UniqueComponentIds::doCheck (const Model& m)
{
  unsigned int n, size, sr, sr_size;

  checkId( m );

  size = m.getNumFunctionDefinitions();
  for (n = 0; n < size; ++n) checkId( *m.getFunctionDefinition(n) );

  size = m.getNumCompartments();
  for (n = 0; n < size; ++n) checkId( *m.getCompartment(n) );

  size = m.getNumSpecies();
  for (n = 0; n < size; ++n) checkId( *m.getSpecies(n) );

  size = m.getNumParameters();
  for (n = 0; n < size; ++n) checkId( *m.getParameter(n) );

  size = m.getNumReactions();
  for (n = 0; n < size; ++n) 
  {
    checkId( *m.getReaction(n) );

    sr_size = m.getReaction(n)->getNumReactants();
    for (sr = 0; sr < sr_size; sr++)
    {
      checkId(*m.getReaction(n)->getReactant(sr));
    }

    sr_size = m.getReaction(n)->getNumProducts();
    for (sr = 0; sr < sr_size; sr++)
    {
      checkId(*m.getReaction(n)->getProduct(sr));
    }

    sr_size = m.getReaction(n)->getNumModifiers();
    for (sr = 0; sr < sr_size; sr++)
    {
      checkId(*m.getReaction(n)->getModifier(sr));
    }

  }

  size = m.getNumEvents();
  for (n = 0; n < size; ++n) checkId( *m.getEvent(n) );

  //size = m.getNumCompartmentTypes();
  //for (n = 0; n < size; ++n) checkId( *m.getCompartmentType(n) );

  //size = m.getNumSpeciesTypes();
  //for (n = 0; n < size; ++n) checkId( *m.getSpeciesType(n) );


  const CompModelPlugin * plug = 
    static_cast <const CompModelPlugin*>(m.getPlugin("comp"));
  if (plug == NULL)
  {
    return;
  }

  size = plug->getNumSubmodels();
  for (n = 0; n < size; ++n) 
  {
    checkId( *(plug->getSubmodel(n)) );

    sr_size = plug->getSubmodel(n)->getNumDeletions();
    for (sr = 0; sr < sr_size; sr++)
    {
      checkId(*(plug->getSubmodel(n)->getDeletion(sr)));
    }

  }

  reset();
}

#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

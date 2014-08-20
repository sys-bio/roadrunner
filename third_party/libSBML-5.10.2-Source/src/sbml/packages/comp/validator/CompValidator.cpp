/**
 * @file    Validator.cpp
 * @brief   Base class for SBML Validators
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <algorithm>
#include <functional>
#include <typeinfo>

#include <sbml/SBMLVisitor.h>

#include <sbml/validator/VConstraint.h>
#include <sbml/validator/Validator.h>
#include <sbml/packages/comp/validator/CompValidator.h>
#include <sbml/packages/comp/common/CompExtensionTypes.h>
#include <sbml/SBMLTypes.h>
/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus


//
// NOTE: ConstraintSet, ValidatorConstraints, and ValidatingVisitor used to
// be in separate .cpp and .h files, but in order to link under MSVC6 (the
// compiler doesn't instantiate templates (i.e. generate code), even when
// told explicitly to do so), the classes needed to be combined into a single
// file.
//


// ----------------------------------------------------------------------
// Apply<T> and ConstraintSet<T>
// ----------------------------------------------------------------------


/*
 * Applies a Constraint<T> to an SBML object of type T.
 */
template <typename T>
struct Apply : public unary_function<TConstraint<T>*, void>
{
  Apply (const Model& m, const T& o) : model(m), object(o) { }


  void operator() (TConstraint<T>* constraint)
  {
    constraint->check(model, object);
  }


  const Model& model;
  const T&     object;
};


template <typename T>
class ConstraintSet
{
public:

   ConstraintSet () { }
  ~ConstraintSet () { }


  /*
   * Adds a Constraint to this ConstraintSet.
   */
  void add (TConstraint<T>* c)
  {
    constraints.push_back(c);
  }

  /*
   * Applies all Constraints in this ConstraintSet to the given SBML object
   * of type T.  Constraint violations are logged to Validator.
   */
  void applyTo (const Model& model, const T& object)
  {
    for_each(constraints.begin(), constraints.end(), Apply<T>(model, object));
  }

  /*
   * @return true if this ConstraintSet is empty, false otherwise.
   */
  bool empty () const
  {
    return constraints.empty();
  }


protected:

  std::list< TConstraint<T>* > constraints;
};



// ----------------------------------------------------------------------




// ----------------------------------------------------------------------
// ValidatorConstraints
// ----------------------------------------------------------------------

/*
 * ValidatorConstraints maintain a separate list of constraints for each
 * SBML type.  This is done so that constraints may be applied efficiently
 * during the validation process.
 */
struct CompValidatorConstraints
{
  ConstraintSet<SBMLDocument>             mSBMLDocument;
  ConstraintSet<Model>                    mModel;
  ConstraintSet<Port>                     mPort;
  ConstraintSet<Submodel>                 mSubmodel;
  ConstraintSet<Deletion>                 mDeletion;
  ConstraintSet<ReplacedElement>          mReplacedElement;
  ConstraintSet<ReplacedBy>               mReplacedBy;
  ConstraintSet<SBaseRef>                 mSBaseRef;
  ConstraintSet<ModelDefinition>          mModelDefinition;
  ConstraintSet<ExternalModelDefinition>  mExtModelDefinition;

  map<VConstraint*,bool> ptrMap;

  ~CompValidatorConstraints ();
  void add (VConstraint* c);
};

/*
 * Deletes constraints (TConstraint(T>*) which are stored in lists
 * (ConstraintSet<T>) of this struct.
 * Since the same pointer values could be stored in different lists
 * (e.g., TConstraint<SimpleSpeciesReference>* is stored in both
 * ConstraintSet<SimpleSpeciesReference> and
 * ConstraintSet<ModifierSimpleSpeciesReference>), a pointer map is used for
 * avoiding segmentation fault caused by deleting the same pointer twice.
 */
CompValidatorConstraints::~CompValidatorConstraints ()
{
  map<VConstraint*,bool>::iterator it = ptrMap.begin();

  while(it != ptrMap.end())
  {
     if(it->second) delete it->first;
     ++it;
  }
}


/*
 * Adds the given Contraint to the appropriate ConstraintSet.
 */
void
CompValidatorConstraints::add (VConstraint* c)
{
  if (c == NULL) return;

  ptrMap.insert(pair<VConstraint*,bool>(c,true));

  if (dynamic_cast< TConstraint<SBMLDocument>* >(c) != NULL)
  {
    mSBMLDocument.add( static_cast< TConstraint<SBMLDocument>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Model>* >(c) != NULL)
  {
    mModel.add( static_cast< TConstraint<Model>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Port>* >(c) != NULL)
  {
    mPort.add( static_cast< TConstraint<Port>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Submodel>* >(c) != NULL)
  {
    mSubmodel.add( static_cast< TConstraint<Submodel>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Deletion>* >(c) != NULL)
  {
    mDeletion.add( static_cast< TConstraint<Deletion>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<ReplacedElement>* >(c) != NULL)
  {
    mReplacedElement.add( static_cast< TConstraint<ReplacedElement>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<ReplacedBy>* >(c) != NULL)
  {
    mReplacedBy.add( static_cast< TConstraint<ReplacedBy>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<SBaseRef>* >(c) != NULL)
  {
    mSBaseRef.add( static_cast< TConstraint<SBaseRef>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<ModelDefinition>* >(c) != NULL)
  {
    mModelDefinition.add( static_cast< TConstraint<ModelDefinition>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<ExternalModelDefinition>* >(c) != NULL)
  {
    mExtModelDefinition.add( static_cast< TConstraint<ExternalModelDefinition>* >(c) );
    return;
  }
}

// ----------------------------------------------------------------------




// ----------------------------------------------------------------------
// ValidatingVisitor
// ----------------------------------------------------------------------


/*
 * An SBMLVisitor visits each object in an SBML object tree, calling the
 * appropriate visit() method for the object visited.
 *
 * A ValidatingVisitor overrides each visit method to validate the given
 * SBML object.
 */
class CompValidatingVisitor: public SBMLVisitor
{
public:

  using SBMLVisitor::visit;
  CompValidatingVisitor (CompValidator& v, const Model& m) : v(v), m(m) { }

  virtual bool visit (const Port &x)
  {
    v.mCompConstraints->mPort.applyTo(m, x);
    return !v.mCompConstraints->mPort.empty();
  }

  virtual bool visit (const Submodel &x)
  {
    v.mCompConstraints->mSubmodel.applyTo(m, x);
    return !v.mCompConstraints->mSubmodel.empty();
  }

  virtual bool visit (const Deletion &x)
  {
    v.mCompConstraints->mDeletion.applyTo(m, x);
    return !v.mCompConstraints->mDeletion.empty();
  }

  virtual bool visit (const ReplacedElement &x)
  {
    v.mCompConstraints->mReplacedElement.applyTo(m, x);
    return !v.mCompConstraints->mReplacedElement.empty();
  }

  virtual bool visit (const ReplacedBy &x)
  {
    v.mCompConstraints->mReplacedBy.applyTo(m, x);
    return !v.mCompConstraints->mReplacedBy.empty();
  }

  virtual bool visit (const SBaseRef &x)
  {
    v.mCompConstraints->mSBaseRef.applyTo(m, x);
    return !v.mCompConstraints->mSBaseRef.empty();
  }

  virtual bool visit (const ModelDefinition &x)
  {
    v.mCompConstraints->mModelDefinition.applyTo(m, x);
    return !v.mCompConstraints->mModelDefinition.empty();
  }

  virtual bool visit (const ExternalModelDefinition &x)
  {
    v.mCompConstraints->mExtModelDefinition.applyTo(m, x);
    return !v.mCompConstraints->mExtModelDefinition.empty();
  }

  virtual void visit (const Model &x)
  {
    v.mCompConstraints->mModel.applyTo(m, x);
  }

  virtual bool visit (const SBase &x)
  {
    if (&x == NULL || x.getPackageName() != "comp")
    {
      return SBMLVisitor::visit(x);      
    }

    int code = x.getTypeCode(); 

    const ListOf* list = dynamic_cast<const ListOf*>(&x);

    if (list != NULL) 
    {
      return SBMLVisitor::visit(x);
    }
    else
    {
      if (code == SBML_COMP_SUBMODEL)
      {
        return visit((const Submodel&)x);
      } 
      else if (code == SBML_COMP_MODELDEFINITION)
      {
        return visit((const ModelDefinition&)x);
      } 
      else if (code == SBML_COMP_EXTERNALMODELDEFINITION)
      {
        return visit((const ExternalModelDefinition&)x);
      } 
      else if (code == SBML_COMP_DELETION)
      {
        return visit((const Deletion&)x);
      } 
      else if (code == SBML_COMP_REPLACEDELEMENT)
      {
        return visit((const ReplacedElement&)x);
      } 
      else if (code == SBML_COMP_REPLACEDBY)
      {
        return visit((const ReplacedBy&)x);
      } 
      else if (code == SBML_COMP_SBASEREF)
      {
        return visit((const SBaseRef&)x);
      } 
      else if (code == SBML_COMP_PORT)
      {
        return visit((const Port&)x);
      } 
      else 
      {
        return SBMLVisitor::visit(x);
      } 
    }
  }


protected:

  CompValidator&   v;
  const Model& m;
};


// ----------------------------------------------------------------------




// ----------------------------------------------------------------------
// Validator
// ----------------------------------------------------------------------


CompValidator::CompValidator (const SBMLErrorCategory_t category):
  Validator(category)
{
  mCompConstraints = new CompValidatorConstraints();
}


CompValidator::~CompValidator ()
{
  delete mCompConstraints;
}


/*
 * Adds the given Contraint to this validator.
 */
void
CompValidator::addConstraint (VConstraint* c)
{
  mCompConstraints->add(c);
}


/*
 * Clears the Validator's list of failures.
 *
 * If you are validating multiple SBML documents with the same Validator,
 * call this method after you have processed the list of failures from the
 * last Validation run and before validating the next document.
 */
//void
//CompValidator::clearFailures ()
//{
//  mFailures.clear();
//}


/*
 * @return the category covered by this Validator.
 */
//const unsigned int
//CompValidator::getCategory () const
//{
//  return mCategory;
//}


/*
 * @return a list of failures logged during validation.
 */
//const std::list<SBMLError>&
//CompValidator::getFailures () const
//{
//  return mFailures;
//}


#define ACCEPT_COMP(x,vv)\
{\
   CompSBasePlugin *plugin = static_cast <CompSBasePlugin *>((x)->getPlugin("comp"));\
   if (plugin != NULL) plugin->accept(vv);\
}

/*
 * Adds the given failure to this list of Validators failures.
 */
//void
//CompValidator::logFailure (const SBMLError& msg)
//{
//  if (&msg == NULL) return;
//  mFailures.push_back(msg);
//}

/*
 * Validates the given SBMLDocument.  Failures logged during
 * validation may be retrieved via <code>getFailures()</code>.
 *
 * @return the number of validation errors that occurred.
 */
unsigned int
CompValidator::validate (const SBMLDocument& d)
{
  if (&d == NULL) return 0;

  Model* m = const_cast<SBMLDocument&>(d).getModel();

  if (m != NULL)
  {
    CompValidatingVisitor vv(*this, *m);

    // validate via the plugins for the package
    const CompSBMLDocumentPlugin* csdp = 
      static_cast <const CompSBMLDocumentPlugin *>(d.getPlugin("comp"));
    
    if (csdp != NULL) 
    {
      csdp->accept(vv);
    }

    CompModelPlugin* cmp = 
      static_cast <CompModelPlugin *>(m->getPlugin("comp"));
    
    if (cmp != NULL) 
    {
      cmp->accept(vv);
    }
    

    unsigned int i = 0;
    unsigned int j;

    for (i = 0; i < m->getNumParameters(); i++)
    {
      ACCEPT_COMP(m->getParameter(i),vv);
    }
    for (i = 0; i < m->getNumCompartments(); i++)
    {
      ACCEPT_COMP(m->getCompartment(i),vv);
    }
    for (i = 0; i < m->getNumSpecies(); i++)
    {
      ACCEPT_COMP(m->getSpecies(i),vv);
    }
    for (i = 0; i < m->getNumFunctionDefinitions(); i++)
    {
      ACCEPT_COMP(m->getFunctionDefinition(i),vv);
    }
    for (i = 0; i < m->getNumUnitDefinitions(); i++)
    {
      ACCEPT_COMP(m->getUnitDefinition(i),vv);
    }
    for (i = 0; i < m->getNumRules(); i++)
    {
      ACCEPT_COMP(m->getRule(i),vv);
    }
    for (i = 0; i < m->getNumInitialAssignments(); i++)
    {
      ACCEPT_COMP(m->getInitialAssignment(i),vv);
    }
    for (i = 0; i < m->getNumConstraints(); i++)
    {
      ACCEPT_COMP(m->getConstraint(i),vv);
    }
    for (i = 0; i < m->getNumReactions(); i++)
    {
      Reaction * r = m->getReaction(i);
      ACCEPT_COMP(r,vv);
      for (j = 0; j < r->getNumReactants(); j++)
      {
        ACCEPT_COMP(r->getReactant(j),vv);
      }
      for (j = 0; j < r->getNumProducts(); j++)
      {
        ACCEPT_COMP(r->getProduct(j),vv);
      }
      for (j = 0; j < r->getNumModifiers(); j++)
      {
        ACCEPT_COMP(r->getModifier(j),vv);
      }
      KineticLaw * kl = r->getKineticLaw();
      if (kl != NULL)
      {
        ACCEPT_COMP(kl,vv);

        for (j = 0; j < kl->getNumLocalParameters(); j++)
        {
          ACCEPT_COMP(kl->getLocalParameter(j),vv);
        }
      }
    }
    for (i = 0; i < m->getNumEvents(); i++)
    {
      Event * e = m->getEvent(i);
      ACCEPT_COMP(e,vv);

      for (j = 0; j < e->getNumEventAssignments(); j++)
      {
        ACCEPT_COMP(e->getEventAssignment(j),vv);        
      }

      if (e->isSetTrigger() == true)
      {
        ACCEPT_COMP(e->getTrigger(),vv);
      }

      if (e->isSetDelay() == true)
      {
        ACCEPT_COMP(e->getDelay(),vv);
      }

      if (e->isSetPriority() == true)
      {
        ACCEPT_COMP(e->getPriority(),vv);
      }
    }
  }


  return (unsigned int)mFailures.size();
}


/*
 * Validates the given SBMLDocument.  Failures logged during
 * validation may be retrieved via <code>getFailures()</code>.
 *
 * @return the number of validation errors that occurred.
 */
unsigned int
CompValidator::validate (const std::string& filename)
{
  if (&filename == NULL) return 0;

  SBMLReader    reader;
  SBMLDocument& d = *reader.readSBML(filename);


  for (unsigned int n = 0; n < d.getNumErrors(); ++n)
  {
    logFailure( *d.getError(n) );
  }

  return validate(d);
}

#endif  /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

// ----------------------------------------------------------------------


/**
 * @file    QualValidator.cpp
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <sbml/validator/VConstraint.h>
#include <sbml/packages/qual/validator/QualValidator.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <sbml/SBMLReader.h>

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
struct QualValidatorConstraints
{
  ConstraintSet<SBMLDocument>             mSBMLDocument;
  ConstraintSet<Model>                    mModel;
  ConstraintSet<QualitativeSpecies>       mQualitativeSpecies;
  ConstraintSet<Transition>               mTransition;
  ConstraintSet<Input>                    mInput;
  ConstraintSet<Output>                   mOutput;
  ConstraintSet<FunctionTerm>             mFunctionTerm;
  ConstraintSet<DefaultTerm>              mDefaultTerm;
  ConstraintSet<ListOfFunctionTerms>      mListOfFunctionTerms;

  map<VConstraint*,bool> ptrMap;

  ~QualValidatorConstraints ();
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
QualValidatorConstraints::~QualValidatorConstraints ()
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
QualValidatorConstraints::add (VConstraint* c)
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

  if (dynamic_cast< TConstraint<QualitativeSpecies>* >(c) != NULL)
  {
    mQualitativeSpecies.add( static_cast< TConstraint<QualitativeSpecies>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Transition>* >(c) != NULL)
  {
    mTransition.add( static_cast< TConstraint<Transition>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Input>* >(c) != NULL)
  {
    mInput.add( static_cast< TConstraint<Input>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<Output>* >(c) != NULL)
  {
    mOutput.add( static_cast< TConstraint<Output>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<FunctionTerm>* >(c) != NULL)
  {
    mFunctionTerm.add( static_cast< TConstraint<FunctionTerm>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<DefaultTerm>* >(c) != NULL)
  {
    mDefaultTerm.add( static_cast< TConstraint<DefaultTerm>* >(c) );
    return;
  }

  if (dynamic_cast< TConstraint<ListOfFunctionTerms>* >(c) != NULL)
  {
    mListOfFunctionTerms.add( static_cast< TConstraint<ListOfFunctionTerms>* >(c) );
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
class QualValidatingVisitor: public SBMLVisitor
{
public:

  using SBMLVisitor::visit;

  QualValidatingVisitor (QualValidator& v, const Model& m) : v(v), m(m) { }

  bool visit (const QualitativeSpecies &x)
  {
    v.mQualConstraints->mQualitativeSpecies.applyTo(m, x);
    return !v.mQualConstraints->mQualitativeSpecies.empty();
  }

  bool visit (const Transition &x)
  {
    v.mQualConstraints->mTransition.applyTo(m, x);
    return !v.mQualConstraints->mTransition.empty();
  }

  bool visit (const Input &x)
  {
    v.mQualConstraints->mInput.applyTo(m, x);
    return !v.mQualConstraints->mInput.empty();
  }

  bool visit (const Output &x)
  {
    v.mQualConstraints->mOutput.applyTo(m, x);
    return !v.mQualConstraints->mOutput.empty();
  }

  bool visit (const FunctionTerm &x)
  {
    v.mQualConstraints->mFunctionTerm.applyTo(m, x);
    return !v.mQualConstraints->mFunctionTerm.empty();
  }

  bool visit (const DefaultTerm &x)
  {
    v.mQualConstraints->mDefaultTerm.applyTo(m, x);
    return !v.mQualConstraints->mDefaultTerm.empty();
  }

  bool visit (const ListOfFunctionTerms &x)
  {
    v.mQualConstraints->mListOfFunctionTerms.applyTo(m, x);
    return !v.mQualConstraints->mListOfFunctionTerms.empty();
  }

  virtual void visit (const Model &x)
  {
    v.mQualConstraints->mModel.applyTo(m, x);
  }

  virtual bool visit (const SBase &x)
  {
    if(&x == NULL || x.getPackageName() != "qual")
    {
      return SBMLVisitor::visit(x);
    }

    int code = x.getTypeCode();

    const ListOf* list = dynamic_cast<const ListOf*>(&x);

    if (list != NULL)
    {
      code = list->getItemTypeCode();

      if (code == SBML_QUAL_FUNCTION_TERM)
      {
        return visit((const ListOfFunctionTerms&)x);
      }
      else
      {
        return SBMLVisitor::visit(x);
      }
    }
    else
    {
      if (code == SBML_QUAL_QUALITATIVE_SPECIES)
      {
        return visit((const QualitativeSpecies&)x);
      }
      else if (code == SBML_QUAL_TRANSITION)
      {
        return visit((const Transition&)x);
      }
      else if (code == SBML_QUAL_INPUT)
      {
        return visit((const Input&)x);
      }
      else if (code == SBML_QUAL_OUTPUT)
      {
        return visit((const Output&)x);
      }
      else if (code == SBML_QUAL_FUNCTION_TERM)
      {
        return visit((const FunctionTerm&)x);
      }
      else if (code == SBML_QUAL_DEFAULT_TERM)
      {
        return visit((const DefaultTerm&)x);
      }
      else 
      {
        return SBMLVisitor::visit(x);
      } 
    }
  }

protected:

  QualValidator&   v;
  const Model& m;
};


// ----------------------------------------------------------------------




// ----------------------------------------------------------------------
// Validator
// ----------------------------------------------------------------------


QualValidator::QualValidator (const SBMLErrorCategory_t category):
  Validator(category)
{
  mQualConstraints = new QualValidatorConstraints();
}


QualValidator::~QualValidator ()
{
  delete mQualConstraints;
}


/*
 * Adds the given Contraint to this validator.
 */
void
QualValidator::addConstraint (VConstraint* c)
{
  mQualConstraints->add(c);
}


/*
 * Validates the given SBMLDocument.  Failures logged during
 * validation may be retrieved via <code>getFailures()</code>.
 *
 * @return the number of validation errors that occurred.
 */
unsigned int
QualValidator::validate (const SBMLDocument& d)
{
  if (&d == NULL) return 0;

  const Model* m = d.getModel();

  if (m != NULL)
  {
    QualValidatingVisitor vv(*this, *m);

    const QualModelPlugin* plugin = 
      static_cast <const QualModelPlugin *> (m->getPlugin("qual"));
      
    if (plugin != NULL)
    {
      plugin->accept(vv);
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
QualValidator::validate (const std::string& filename)
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


#endif /* __cplusplus */
LIBSBML_CPP_NAMESPACE_END

// ----------------------------------------------------------------------


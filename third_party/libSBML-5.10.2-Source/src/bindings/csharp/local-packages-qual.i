/**
 * @file    local-packages-qual.i
 * @brief   cs-specific SWIG directives for wrapping libSBML layout API this file 
 *          has been adapted from the SWIG java bindings written by 
 * 	    Ben Bornstein and Akiya Jouraku
 * @author  Frank Bergmann (fbergman@u.washington.edu)
 * @author  Akiya Jouraku
 *
 * $Id: local.i 10190 2009-09-23 16:03:35Z ajouraku $
 * $URL: https://sbml.svn.sourceforge.net/svnroot/sbml/trunk/libsbml/src/bindings/csharp/local.i $
 */

/*
 *This file is part of libSBML.  Please visit http://sbml.org for more
 *information about SBML, and the latest version of libSBML.
 *
 *Copyright (C) 2013-2014 jointly by the following organizations:
 *    1. California Institute of Technology, Pasadena, CA, USA
 *    2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *    3. University of Heidelberg, Heidelberg, Germany
 *
 *Copyright 2008 California Institute of Technology.
 *
 *This library is free software; you can redistribute it and/or modify it
 *under the terms of the GNU Lesser General Public License as published by
 *the Free Software Foundation.  A copy of the license agreement is provided
 *in the file named "LICENSE.txt" included with this software distribution
 *and also available online as http://sbml.org/software/libsbml/license.html
 */

#ifdef USE_QUAL

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the qual package extension
 */
%typemap(cscode) QualExtension
%{
	public override SBasePlugin DowncastSBasePlugin(IntPtr cPtr, bool owner)
	{
		if (cPtr.Equals(IntPtr.Zero)) return null;
		
		SBasePlugin sbp = new SBasePlugin(cPtr, false);
		SBase sb = sbp.getParentSBMLObject();

		switch( sb.getTypeCode() )
		{
			case (int) libsbml.SBML_MODEL:
				return new QualModelPlugin(cPtr,owner);
			default:
				return new SBasePlugin(cPtr,owner);
		}
	}

	public override SBase DowncastSBase(IntPtr cPtr, bool owner)
	{
		if (cPtr.Equals(IntPtr.Zero)) return null;
		
		SBase sb = new SBase(cPtr, false);
		switch( sb.getTypeCode() )
		{
			case (int) libsbml.SBML_LIST_OF:
			     string name = sb.getElementName();
		       if(name =="listOfFunctionTerms")
			     {
		           return new ListOfFunctionTerms(cPtr, owner);
           }
		       else if(name =="listOfInputs")
			     {
		           return new ListOfInputs(cPtr, owner);
           }
		       else if(name =="listOfOutputs")
			     {
		           return new ListOfOutputs(cPtr, owner);
           }
		       else if(name =="listOfQualitativeSpecies")
			     {
		           return new ListOfQualitativeSpecies(cPtr, owner);
           }
		       else if(name =="listOfTransitions")
			     {
		           return new ListOfTransitions(cPtr, owner);
           }
		       
		       return new ListOf(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_QUALITATIVE_SPECIES:
				return new QualitativeSpecies(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_TRANSITION:
				return new Transition(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_INPUT:
				return new Input(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_OUTPUT:
				return new Output(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_FUNCTION_TERM:
				return new FunctionTerm(cPtr, owner);
				
			case (int) libsbml.SBML_QUAL_DEFAULT_TERM:
				return new DefaultTerm(cPtr, owner);
				
			default:
				return new SBase(cPtr, owner);
		}
	}
%}

COVARIANT_RTYPE_CLONE(QualExtension)
COVARIANT_RTYPE_CLONE(Transition)
COVARIANT_RTYPE_CLONE(FunctionTerm)
COVARIANT_RTYPE_CLONE(DefaultTerm)
COVARIANT_RTYPE_CLONE(QualitativeSpecies)
COVARIANT_RTYPE_CLONE(Input)
COVARIANT_RTYPE_CLONE(Output)
COVARIANT_RTYPE_CLONE(ListOfFunctionTerms)
COVARIANT_RTYPE_CLONE(ListOfInputs)
COVARIANT_RTYPE_CLONE(ListOfOutputs)
COVARIANT_RTYPE_CLONE(ListOfQualitativeSpecies)
COVARIANT_RTYPE_CLONE(ListOfTransitions)

COVARIANT_RTYPE_LISTOF_GET_REMOVE(Transition)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(FunctionTerm)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(Input)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(Output)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(QualitativeSpecies)

%define COVARIANT_RTYPE_ISSETID(_CNAME_)
%typemap(cstype) _CNAME_* _CNAME_::isSetId  "_CNAME_"
%csmethodmodifiers _CNAME_::isSetId  "public new"
%enddef

COVARIANT_RTYPE_ISSETID(Input)

SBMLCONSTRUCTOR_EXCEPTION(QualPkgNamespaces)
SBMLCONSTRUCTOR_EXCEPTION(Transition)
SBMLCONSTRUCTOR_EXCEPTION(FunctionTerm)
SBMLCONSTRUCTOR_EXCEPTION(DefaultTerm)
SBMLCONSTRUCTOR_EXCEPTION(QualitativeSpecies)
SBMLCONSTRUCTOR_EXCEPTION(Input)
SBMLCONSTRUCTOR_EXCEPTION(Output)
SBMLCONSTRUCTOR_EXCEPTION(ListOfFunctionTerms)
SBMLCONSTRUCTOR_EXCEPTION(ListOfInputs)
SBMLCONSTRUCTOR_EXCEPTION(ListOfOutputs)
SBMLCONSTRUCTOR_EXCEPTION(ListOfQualitativeSpecies)
SBMLCONSTRUCTOR_EXCEPTION(ListOfTransitions)

#endif  /* USE_QUAL */


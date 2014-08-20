/**
 * @file    local-packages-comp.i
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

#ifdef USE_COMP

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the layout package extension
 */
%typemap(cscode) CompExtension
%{
	public override SBasePlugin DowncastSBasePlugin(IntPtr cPtr, bool owner)
	{
		if (cPtr.Equals(IntPtr.Zero)) return null;
		
		SBasePlugin sbp = new SBasePlugin(cPtr, false);
		SBase sb = sbp.getParentSBMLObject();

		if (sb == null) 
		{
			return new SBasePlugin(cPtr,owner);
		}
		
		switch( sb.getTypeCode() )
		{					
			case (int) libsbml.SBML_DOCUMENT:
				return new CompSBMLDocumentPlugin(cPtr,owner);
    
			case (int) libsbml.SBML_MODEL:
			case (int) libsbml.SBML_COMP_MODELDEFINITION:
				return new CompModelPlugin(cPtr, owner);
			case (int) libsbml.SBML_COMPARTMENT:
			case (int) libsbml.SBML_COMPARTMENT_TYPE:
			case (int) libsbml.SBML_CONSTRAINT:
			case (int) libsbml.SBML_EVENT:
			case (int) libsbml.SBML_EVENT_ASSIGNMENT:
			case (int) libsbml.SBML_FUNCTION_DEFINITION:
			case (int) libsbml.SBML_INITIAL_ASSIGNMENT:
			case (int) libsbml.SBML_KINETIC_LAW:
			case (int) libsbml.SBML_LIST_OF:
			case (int) libsbml.SBML_PARAMETER:
			case (int) libsbml.SBML_REACTION:
			case (int) libsbml.SBML_RULE:
			case (int) libsbml.SBML_SPECIES:
			case (int) libsbml.SBML_SPECIES_REFERENCE:
			case (int) libsbml.SBML_SPECIES_TYPE:
			case (int) libsbml.SBML_MODIFIER_SPECIES_REFERENCE:
			case (int) libsbml.SBML_UNIT_DEFINITION:
			case (int) libsbml.SBML_UNIT:
			case (int) libsbml.SBML_ALGEBRAIC_RULE:
			case (int) libsbml.SBML_ASSIGNMENT_RULE:
			case (int) libsbml.SBML_RATE_RULE:
			case (int) libsbml.SBML_SPECIES_CONCENTRATION_RULE:
			case (int) libsbml.SBML_COMPARTMENT_VOLUME_RULE:
			case (int) libsbml.SBML_PARAMETER_RULE:
			case (int) libsbml.SBML_TRIGGER:
			case (int) libsbml.SBML_DELAY:
			case (int) libsbml.SBML_STOICHIOMETRY_MATH:
			case (int) libsbml.SBML_LOCAL_PARAMETER:
			case (int) libsbml.SBML_COMP_SUBMODEL:
			case (int) libsbml.SBML_COMP_SBASEREF:
			case (int) libsbml.SBML_COMP_REPLACEDELEMENT:
			case (int) libsbml.SBML_COMP_REPLACEDBY:
			case (int) libsbml.SBML_COMP_PORT:
			case (int) libsbml.SBML_COMP_DELETION:
			case (int) libsbml.SBML_COMP_EXTERNALMODELDEFINITION:
				return new CompSBasePlugin(cPtr,owner);
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
		         if(name =="listOfDeletions")
			     {
					return new ListOfDeletions(cPtr, owner);
                 }
		         else if(name =="listOfExternalModelDefinitions")
			     {
		            return new ListOfExternalModelDefinitions(cPtr, owner);
                 }
				 else if(name =="listOfModelDefinitions")
			     {
		            return new ListOfModelDefinitions(cPtr, owner);
                 }
				 else if(name =="listOfPorts")
			     {
		            return new ListOfPorts(cPtr, owner);
                 }
				 else if(name =="listOfReplacedElements")
			     {
		            return new ListOfReplacedElements(cPtr, owner);
                 }
				 else if(name =="listOfSubmodels")
			     {
		            return new ListOfSubmodels(cPtr, owner);
                 }
		         return new ListOf(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_DELETION:
				return new Deletion(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_MODELDEFINITION:
				return new ModelDefinition(cPtr, owner);

			case (int) libsbml.SBML_COMP_EXTERNALMODELDEFINITION:
				return new ExternalModelDefinition(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_PORT:
				return new Port(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_REPLACEDELEMENT:
				return new ReplacedElement(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_REPLACEDBY:
				return new ReplacedBy(cPtr, owner);

			case (int) libsbml.SBML_COMP_SBASEREF:
				return new SBaseRef(cPtr, owner);
				
			case (int) libsbml.SBML_COMP_SUBMODEL:
				return new Submodel(cPtr, owner);
				
			default:
				return new SBase(cPtr, owner);
		}
	}
%}

SBMLCONSTRUCTOR_EXCEPTION(CompPkgNamespaces)
SBMLCONSTRUCTOR_EXCEPTION(SBaseRef)
SBMLCONSTRUCTOR_EXCEPTION(Deletion)
SBMLCONSTRUCTOR_EXCEPTION(ExternalModelDefinition)
SBMLCONSTRUCTOR_EXCEPTION(ListOfDeletions)
SBMLCONSTRUCTOR_EXCEPTION(ListOfExternalModelDefinitions)
SBMLCONSTRUCTOR_EXCEPTION(ListOfModelDefinitions)
SBMLCONSTRUCTOR_EXCEPTION(ListOfPorts)
SBMLCONSTRUCTOR_EXCEPTION(ListOfReplacedElements)
SBMLCONSTRUCTOR_EXCEPTION(ListOfSubmodels)
SBMLCONSTRUCTOR_EXCEPTION(ModelDefinition)
SBMLCONSTRUCTOR_EXCEPTION(Port)
SBMLCONSTRUCTOR_EXCEPTION(ReplacedBy)
SBMLCONSTRUCTOR_EXCEPTION(ReplacedElement)
SBMLCONSTRUCTOR_EXCEPTION(Submodel)


%define COVARIANT_RTYPE_RENAMESIDREFS(_CNAME_)
%typemap(cstype) _CNAME_* _CNAME_::renameSIdRefs  "_CNAME_"
%csmethodmodifiers _CNAME_::renameSIdRefs  "public new"
%enddef

COVARIANT_RTYPE_RENAMESIDREFS(ReplacedElement)
COVARIANT_RTYPE_RENAMESIDREFS(Port)

%define COVARIANT_RTYPE_RENAMEUNITSIDREFS(_CNAME_)
%typemap(cstype) _CNAME_* _CNAME_::renameUnitSIdRefs  "_CNAME_"
%csmethodmodifiers _CNAME_::renameUnitSIdRefs  "public new"
%enddef

COVARIANT_RTYPE_RENAMEUNITSIDREFS(Port)

%define COVARIANT_RTYPE_FUNCTION(_CNAME_, _FNAME_)
%typemap(cstype) _CNAME_* _CNAME_::_FNAME_  "_CNAME_"
%csmethodmodifiers _CNAME_::_FNAME_  "public new"
%enddef

COVARIANT_RTYPE_FUNCTION(Port, unsetName)
COVARIANT_RTYPE_FUNCTION(Port, setName)
COVARIANT_RTYPE_FUNCTION(Port, isSetName)
COVARIANT_RTYPE_FUNCTION(Port, getId)

%define COVARIANT_RTYPE_RENAMEMETAIDREFS(_CNAME_)
%typemap(cstype) _CNAME_* _CNAME_::renameMetaIdRefs  "_CNAME_"
%csmethodmodifiers _CNAME_::renameMetaIdRefs  "public new"
%enddef

COVARIANT_RTYPE_RENAMEMETAIDREFS(Port)

%define COVARIANT_REMOVEFROMPARENTANDDELETE(_CNAME_)
%typemap(cstype) string   _CNAME_ ## ::removeFromParentAndDelete  "_CNAME_"
%csmethodmodifiers  _CNAME_ ## ::removeFromParentAndDelete  "public new"
%enddef

COVARIANT_REMOVEFROMPARENTANDDELETE(ReplacedBy)
COVARIANT_REMOVEFROMPARENTANDDELETE(ModelDefinition)

#endif



/**
 * @file    local-layout.i
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

#ifdef USE_FBC

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the fbc package extension
 */
%typemap(cscode) FbcExtension
%{
	public override SBasePlugin DowncastSBasePlugin(IntPtr cPtr, bool owner)
	{
		if (cPtr.Equals(IntPtr.Zero)) return null;
		
		SBasePlugin sbp = new SBasePlugin(cPtr, false);
		SBase sb = sbp.getParentSBMLObject();

		switch( sb.getTypeCode() )
		{
			case (int) libsbml.SBML_MODEL:
				return new FbcModelPlugin(cPtr,owner);
      case (int) libsbml.SBML_SPECIES:
        return new FbcSpeciesPlugin(cPtr, owner);
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
		         if(name =="listOfFluxBounds")
			     {
					return new ListOfFluxBounds(cPtr, owner);
                 }
		         else if(name =="listOfFluxes")
			     {
		            return new ListOfFluxObjectives(cPtr, owner);
                 }
				 else if(name =="listOfGeneAssociations")
			     {
		            return new ListOfGeneAssociations(cPtr, owner);
                 }
				 else if(name =="listOfObjectives")
			     {
		            return new ListOfObjectives(cPtr, owner);
                 }
		         return new ListOf(cPtr, owner);
				
			case (int) libsbml.SBML_FBC_ASSOCIATION:
				return new Association(cPtr, owner);
				
			case (int) libsbml.SBML_FBC_FLUXBOUND:
				return new FluxBound(cPtr, owner);

			case (int) libsbml.SBML_FBC_FLUXOBJECTIVE:
				return new FluxObjective(cPtr, owner);
				
			case (int) libsbml.SBML_FBC_GENEASSOCIATION:
				return new GeneAssociation(cPtr, owner);
				
			case (int) libsbml.SBML_FBC_OBJECTIVE:
				return new Objective(cPtr, owner);
				
			default:
				return new SBase(cPtr, owner);
		}
	}
%}

COVARIANT_RTYPE_CLONE(FbcExtension)
COVARIANT_RTYPE_CLONE(Association)
COVARIANT_RTYPE_CLONE(GeneAssociation)
COVARIANT_RTYPE_CLONE(Objective)
COVARIANT_RTYPE_CLONE(FluxObjective)
COVARIANT_RTYPE_CLONE(FluxBound)
COVARIANT_RTYPE_CLONE(ListOfFluxBounds)
COVARIANT_RTYPE_CLONE(ListOfFluxObjectives)
COVARIANT_RTYPE_CLONE(ListOfGeneAssociations)
COVARIANT_RTYPE_CLONE(ListOfObjectives)

COVARIANT_RTYPE_LISTOF_GET_REMOVE(Association)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(GeneAssociation)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(Objective)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(FluxObjective)
COVARIANT_RTYPE_LISTOF_GET_REMOVE(FluxBound)

SBMLCONSTRUCTOR_EXCEPTION(Association)
SBMLCONSTRUCTOR_EXCEPTION(FbcPkgNamespaces)
SBMLCONSTRUCTOR_EXCEPTION(FluxBound)
SBMLCONSTRUCTOR_EXCEPTION(FluxObjective)
SBMLCONSTRUCTOR_EXCEPTION(GeneAssociation)
SBMLCONSTRUCTOR_EXCEPTION(ListOfFluxBounds)
SBMLCONSTRUCTOR_EXCEPTION(ListOfFluxObjectives)
SBMLCONSTRUCTOR_EXCEPTION(ListOfGeneAssociations)
SBMLCONSTRUCTOR_EXCEPTION(ListOfObjectives)
SBMLCONSTRUCTOR_EXCEPTION(Objective)


COVARIANT_GETID(FluxBound)

#endif  /* USE_FBC */


#ifdef USE_FBC

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the layout package extension
 */
%typemap(javacode) FbcExtension
%{
	public SBasePlugin DowncastSBasePlugin(long cPtr, boolean owner)
	{
		if (cPtr == 0) return null;
		
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
	
	public SBase DowncastSBase(long cPtr, boolean owner)
	{
		if (cPtr == 0) return null;
		
		SBase sb = new SBase(cPtr, false);
		switch( sb.getTypeCode() )
		{
			case (int) libsbml.SBML_LIST_OF:
			     String name = sb.getElementName();
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

#endif  /* USE_FBC */

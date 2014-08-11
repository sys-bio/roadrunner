
#ifdef USE_FBC
else if (pkgName == "fbc")
{
	switch (sb->getTypeCode())
	{
		case SBML_LIST_OF:
			name = sb->getElementName();
			if(name == "listOfFluxes"){
				return SWIGTYPE_p_ListOfFluxObjectives;
			}
			else if(name == "listOfGeneAssociations"){
				return SWIGTYPE_p_ListOfGeneAssociations;
			}
			else if(name == "listOfObjectives"){
				return SWIGTYPE_p_ListOfObjectives;
			}			
			return SWIGTYPE_p_ListOf;				  
			
		case SBML_FBC_ASSOCIATION:
			return SWIGTYPE_p_Association;
		case SBML_FBC_FLUXBOUND:
			return SWIGTYPE_p_FluxBound;
		case SBML_FBC_FLUXOBJECTIVE:
			return SWIGTYPE_p_FluxObjective;
		case SBML_FBC_GENEASSOCIATION:
			return SWIGTYPE_p_GeneAssociation;
		case SBML_FBC_OBJECTIVE:
			return SWIGTYPE_p_Objective;
		default:
			return SWIGTYPE_p_SBase;
	}
}
#endif // USE_FBC				  


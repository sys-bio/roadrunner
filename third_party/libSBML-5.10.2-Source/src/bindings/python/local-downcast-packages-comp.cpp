
#ifdef USE_COMP
else if (pkgName == "comp")
{
	switch (sb->getTypeCode())
	{
		case SBML_LIST_OF:
			name = sb->getElementName();
			if(name =="listOfDeletions")
			{
				return SWIGTYPE_p_ListOfDeletions;
            }
		    else if(name =="listOfExternalModelDefinitions")
			{
		       return SWIGTYPE_p_ListOfExternalModelDefinitions;
            }
			else if(name =="listOfModelDefinitions")
			{
		       return SWIGTYPE_p_ListOfModelDefinitions;
            }
			else if(name =="listOfPorts")
			{
		       return SWIGTYPE_p_ListOfPorts;
            }
			else if(name =="listOfReplacedElements")
			{
		       return SWIGTYPE_p_ListOfReplacedElements;
            }
			else if(name =="listOfSubmodels")
			{
		       return SWIGTYPE_p_ListOfSubmodels;
            }
			return SWIGTYPE_p_ListOf;				  
			
		case SBML_COMP_DELETION:
			return SWIGTYPE_p_Deletion;
			
		case SBML_COMP_MODELDEFINITION:
			return SWIGTYPE_p_ModelDefinition;

		case SBML_COMP_EXTERNALMODELDEFINITION:
			return SWIGTYPE_p_ExternalModelDefinition;
			
		case SBML_COMP_PORT:
			return SWIGTYPE_p_Port;
			
		case SBML_COMP_REPLACEDELEMENT:
			return SWIGTYPE_p_ReplacedElement;
			
		case SBML_COMP_REPLACEDBY:
			return SWIGTYPE_p_ReplacedBy;
				
		case SBML_COMP_SBASEREF:
			return SWIGTYPE_p_SBaseRef;
			
		case SBML_COMP_SUBMODEL:
			return SWIGTYPE_p_Submodel;
			
		default:
			return SWIGTYPE_p_SBase;
				
	}
}
#endif // USE_COMP				  


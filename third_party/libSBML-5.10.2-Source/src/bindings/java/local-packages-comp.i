#ifdef USE_COMP

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the layout package extension
 */
%typemap(javacode) CompExtension
%{
	public SBasePlugin DowncastSBasePlugin(long cPtr, boolean owner)
	{
		if (cPtr == 0) return null;
		
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
			case (int) libsbml.SBML_COMP_EXTERNALMODELDEFINITION:
			case (int) libsbml.SBML_COMP_DELETION:
				return new CompSBasePlugin(cPtr,owner);
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

// mark the clone method as covariant 
COVARIANT_RTYPE_CLONE(ModelDefinition)
// rename to prevent java from getting confused with SBase::getModel
%rename(getModelRef) ExternalModelDefinition::getModel;

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


COVARIANT_RTYPE_CLONE(ListOfSubmodels)
COVARIANT_RTYPE_CLONE(ListOfPorts)
COVARIANT_RTYPE_CLONE(ListOfReplacedElements)
COVARIANT_RTYPE_CLONE(ListOfDeletions)
COVARIANT_RTYPE_CLONE(ListOfModelDefinitions)
COVARIANT_RTYPE_CLONE(ListOfExternalModelDefinitions)

#endif  /* USE_COMP */

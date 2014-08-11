#ifdef USE_QUAL

/**
 * Adds DownCastBase(long cPtr, boolean owner) method for the layout package extension
 */
%typemap(javacode) QualExtension
%{
	public SBasePlugin DowncastSBasePlugin(long cPtr, boolean owner)
	{
		if (cPtr == 0) return null;
		
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
	
	public SBase DowncastSBase(long cPtr, boolean owner)
	{
		if (cPtr == 0) return null;
		
		SBase sb = new SBase(cPtr, false);
		switch( sb.getTypeCode() )
		{
			case (int) libsbml.SBML_LIST_OF:
				String name = sb.getElementName();
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

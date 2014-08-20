
#ifdef USE_QUAL
else if (pkgName == "qual")
{
	switch (sb->getTypeCode())
	{
		case SBML_LIST_OF:
			name = sb->getElementName();
		  if(name =="listOfFunctionTerms")
			{
		      return SWIGTYPE_p_ListOfFunctionTerms;
      }
		  else if(name =="listOfInputs")
			{
		      return SWIGTYPE_p_ListOfInputs;
      }
		  else if(name =="listOfOutputs")
			{
		      return SWIGTYPE_p_ListOfOutputs;
      }
		  else if(name =="listOfQualitativeSpecies")
			{
		      return SWIGTYPE_p_ListOfQualitativeSpecies;
      }
		  else if(name =="listOfTransitions")
			{
		      return SWIGTYPE_p_ListOfTransitions;
      }
			
      return SWIGTYPE_p_ListOf;				  
			
		case SBML_QUAL_QUALITATIVE_SPECIES:
			return SWIGTYPE_p_QualitativeSpecies;
				
		case SBML_QUAL_TRANSITION:
			return SWIGTYPE_p_Transition;
				
		case SBML_QUAL_INPUT:
			return SWIGTYPE_p_Input;
				
		case SBML_QUAL_OUTPUT:
			return SWIGTYPE_p_Output;
				
		case SBML_QUAL_FUNCTION_TERM:
			return SWIGTYPE_p_FunctionTerm;

		case SBML_QUAL_DEFAULT_TERM:
			return SWIGTYPE_p_DefaultTerm;

    default:
			return SWIGTYPE_p_SBase;
	}
}
#endif // USE_QUAL				  


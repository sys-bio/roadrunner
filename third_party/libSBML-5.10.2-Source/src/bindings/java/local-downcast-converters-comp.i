/**
 * casting to most specific SBMLConverter object
 */
 
#ifdef USE_COMP
%pragma(java) modulecode =
%{
	if (conName.equals("SBML Comp Flattening Converter"))
		return new CompFlatteningConverter(cPtr, owner);
%}
#endif /* USE_COMP */
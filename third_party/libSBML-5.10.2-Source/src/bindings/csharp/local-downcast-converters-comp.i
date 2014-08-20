/**
 * casting to most specific SBMLConverter object
 */
 
#ifdef USE_COMP
%pragma(csharp) modulecode =
%{
	if (conName == "SBML Comp Flattening Converter")
		return new CompFlatteningConverter(cPtr, owner);
%}
#endif /* USE_COMP */
/**
 * casting to most specific SBMLConverter object
 */
 
#ifdef USE_FBC
%pragma(java) modulecode =
%{
	if (conName.equals("SBML FBC to COBRA Converter"))
		return new FbcToCobraConverter(cPtr, owner);
	if (conName.equals("SBML COBRA to FBC Converter"))
		return new CobraToFbcConverter(cPtr, owner);
%}
#endif /* USE_FBC*/

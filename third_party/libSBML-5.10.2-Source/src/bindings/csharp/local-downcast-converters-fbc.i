/**
 * casting to most specific SBMLConverter object
 */
 
#ifdef USE_FBC
%pragma(csharp) modulecode =
%{
	if (conName == "SBML FBC to COBRA Converter")
		return new FbcToCobraConverter(cPtr, owner);
	if (conName == "SBML COBRA to FBC Converter")
		return new CobraToFbcConverter(cPtr, owner);
%}
#endif /* USE_FBC*/

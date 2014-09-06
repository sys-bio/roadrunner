/**
 * casting to most specific SBMLExtension object
 */
 
#ifdef USE_FBC
%pragma(csharp) modulecode =
%{
		if (pkgName == "fbc")
			return new FbcExtension(cPtr, owner);
%}
#endif /* USE_FBC*/

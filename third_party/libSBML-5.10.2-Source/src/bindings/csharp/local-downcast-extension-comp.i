/**
 * casting to most specific SBMLExtension object
 */
 
#ifdef USE_COMP
%pragma(csharp) modulecode =
%{
		if (pkgName == "comp")
			return new CompExtension(cPtr, owner);
%}
#endif /* USE_COMP */
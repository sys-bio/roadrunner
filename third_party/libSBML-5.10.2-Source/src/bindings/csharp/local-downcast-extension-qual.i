/**
 * casting to most specific SBMLExtension object
 */
 
#ifdef USE_QUAL
%pragma(csharp) modulecode =
%{
		if (pkgName == "qual")
			return new QualExtension(cPtr, owner);
%}
#endif /* USE_QUAL*/
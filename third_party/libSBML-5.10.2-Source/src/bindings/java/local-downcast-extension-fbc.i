/**
 * casting to most specific SBMLExtension object
 */
 

#ifdef USE_FBC
%pragma(java) modulecode =
%{
    if (pkgName.equals("fbc"))
		return new FbcExtension(cPtr, owner);
	%}
#endif // USE_FBC		


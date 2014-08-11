/**
 * casting to most specific SBMLExtension object
 */
 

#ifdef USE_COMP
%pragma(java) modulecode =
%{
    if (pkgName.equals("comp"))
		return new CompExtension(cPtr, owner);
	%}
#endif // USE_COMP		


/**
 * casting to most specific SBMLExtension object
 */
 

#ifdef USE_QUAL
%pragma(java) modulecode =
%{
    if (pkgName.equals("qual"))
		return new QualExtension(cPtr, owner);
	%}
#endif // USE_QUAL		


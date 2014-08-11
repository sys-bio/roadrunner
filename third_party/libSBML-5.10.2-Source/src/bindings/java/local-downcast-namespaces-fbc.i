/**
 * casting to most specific SBMLNamespaces object
 */


#ifdef USE_FBC
%pragma(java) modulecode =
%{
	
	if (ns.hasURI(FbcExtension.getXmlnsL3V1V1()))
	{
		return new FbcPkgNamespaces(cPtr, owner);
	}
	%}
#endif /* USE_FBC*/

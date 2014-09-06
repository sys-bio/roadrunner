/**
 * casting to most specific SBMLNamespaces object
 */


#ifdef USE_COMP
%pragma(java) modulecode =
%{
	
	if (ns.hasURI(CompExtension.getXmlnsL3V1V1()))
	{
		return new CompPkgNamespaces(cPtr, owner);
	}
	%}
#endif /* USE_COMP*/

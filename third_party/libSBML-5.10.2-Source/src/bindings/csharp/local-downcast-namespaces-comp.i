/**
 * casting to most specific SBMLNamespaces object
 */


#ifdef USE_COMP
%pragma(csharp) modulecode =
%{
	
	if (ns.hasURI(CompExtension.getXmlnsL3V1V1()))
	{
		return new CompPkgNamespaces(cPtr, owner);
	}
%}
#endif /* USE_COMP */

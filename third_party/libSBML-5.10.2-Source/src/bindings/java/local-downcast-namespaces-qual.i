/**
 * casting to most specific SBMLNamespaces object
 */


#ifdef USE_QUAL
%pragma(java) modulecode =
%{
	
	if (ns.hasURI(QualExtension.getXmlnsL3V1V1()))
	{
		return new QualPkgNamespaces(cPtr, owner);
	}
	%}
#endif /* USE_QUAL*/

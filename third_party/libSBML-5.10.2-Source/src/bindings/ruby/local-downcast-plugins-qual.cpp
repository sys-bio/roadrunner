
#ifdef USE_QUAL
if (pkgName == "qual")
{
	if (sb->getTypeCode() == SBML_MODEL)
	{
		return SWIGTYPE_p_QualModelPlugin;
	}
}
#endif // USE_QUAL


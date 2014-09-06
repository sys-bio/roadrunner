
#ifdef USE_FBC
if (pkgName == "fbc")
{
	if (sb->getTypeCode() == SBML_MODEL)
	{
		return SWIGTYPE_p_FbcModelPlugin;
	}
	else if (sb->getTypeCode() == SBML_SPECIES)
	{
		return SWIGTYPE_p_FbcSpeciesPlugin;
	}
}
#endif // USE_FBC


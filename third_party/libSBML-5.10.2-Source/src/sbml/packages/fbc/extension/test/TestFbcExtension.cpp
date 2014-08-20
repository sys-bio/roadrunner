/**
 * @file    TestFbcExtension.cpp
 * @brief   TestFbcExtension unit tests
 * @author  Akiya Jouraku
 *
 * $Id: $
 * $HeadURL: $
 */

#include <limits>

#include <iostream>
#include <check.h>
#include <sbml/conversion/ConversionProperties.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/extension/FbcModelPlugin.h>
#include <sbml/packages/fbc/sbml/Association.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/SBMLTypeCodes.h>
#include <sbml/SBMLReader.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART
extern char *TestDataDirectory;

static FbcExtension* G; 
static FbcPkgNamespaces* GNS;
static string FBC_XMLNS_L3V1V1;
static string CORE_XMLNS_L2V4;
static string FBC_PACKAGE_NAME;

void
FbcExtensionTest_setup (void)
{
  try
  {
    G = new FbcExtension();
    GNS = new FbcPkgNamespaces();
    FBC_PACKAGE_NAME = G->getName();
    FBC_XMLNS_L3V1V1 = GNS->getURI();
    CORE_XMLNS_L2V4 = SBMLNamespaces::getSBMLNamespaceURI(2, 4);
  }
  catch(...)
  {
    fail("Failed to create a FbcExtension object");
  }
}


void
FbcExtensionTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_FbcExtension_getName)
{
  fail_unless(G->getName() == "fbc");
  fail_unless(G->getName() == FBC_PACKAGE_NAME);
}
END_TEST


START_TEST (test_FbcExtension_getURI)
{
  fail_unless(G->getURI(3,1,1) == FBC_XMLNS_L3V1V1);
  fail_unless(G->getURI(2,1,1) == "");
  fail_unless(G->getURI(4,1,1) == "");
}
END_TEST


START_TEST (test_FbcExtension_getLevelVersion)
{
  fail_unless(G->getLevel(FBC_XMLNS_L3V1V1) == 3);
  fail_unless(G->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getLevel("")                          == 0);

  fail_unless(G->getVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(G->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getVersion("")                          == 0);

  fail_unless(G->getPackageVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(G->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getPackageVersion("")                          == 0);
}
END_TEST


START_TEST (test_FbcExtension_getSBMLExtensionNamespaces)
{
  FbcPkgNamespaces *fbcns;
  fbcns = static_cast<FbcPkgNamespaces*>(G->getSBMLExtensionNamespaces(FBC_XMLNS_L3V1V1));

  fail_unless(fbcns->getLevel()          == 3);
  fail_unless(fbcns->getVersion()        == 1);
  fail_unless(fbcns->getPackageVersion() == 1);

  fbcns = static_cast<FbcPkgNamespaces*>(G->getSBMLExtensionNamespaces(""));

  fail_unless(fbcns == NULL);
}
END_TEST


START_TEST(test_FbcExtension_copy)
{
  FbcExtension *g2 = new FbcExtension(*G);

  fail_unless(g2->getName() == "fbc");
  fail_unless(g2->getName() == FBC_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == FBC_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(FBC_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_FbcExtension_assignment)
{
  FbcExtension* g2 = new FbcExtension();

  (*g2) = (*G);

  fail_unless(g2->getName() == "fbc");
  fail_unless(g2->getName() == FBC_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == FBC_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(FBC_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_FbcExtension_clone)
{
  FbcExtension* g2 = G->clone();

  fail_unless(g2->getName() == "fbc");
  fail_unless(g2->getName() == FBC_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == FBC_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(FBC_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_FbcExtension_registry)
{
  const SBMLExtension* sbext = SBMLExtensionRegistry::getInstance().getExtension("fbc");

  fail_unless(sbext != NULL);

  fail_unless(sbext->getName() == "fbc");
  fail_unless(sbext->getName() == FBC_PACKAGE_NAME);

  fail_unless(sbext->getURI(3,1,1) == FBC_XMLNS_L3V1V1);
  fail_unless(sbext->getURI(2,1,1) == "");
  fail_unless(sbext->getURI(4,1,1) == "");

  fail_unless(sbext->getLevel(FBC_XMLNS_L3V1V1) == 3);
  fail_unless(sbext->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getLevel("")                          == 0);

  fail_unless(sbext->getVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(sbext->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getVersion("")                          == 0);

  fail_unless(sbext->getPackageVersion(FBC_XMLNS_L3V1V1) == 1);
  fail_unless(sbext->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getPackageVersion("")                          == 0);
}
END_TEST


START_TEST(test_FbcExtension_typecode)
{
  const SBMLExtension* sbext = SBMLExtensionRegistry::getInstance().getExtension("fbc");

  fail_unless(sbext != NULL);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_OBJECTIVE), "Objective") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_GENEASSOCIATION), "GeneAssociation") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_ASSOCIATION), "Association") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_FLUXBOUND), "FluxBound") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_FLUXOBJECTIVE), "FluxObjective") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_ASSOCIATION-1), "(Unknown SBML Fbc Type)") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_FBC_OBJECTIVE+1), "(Unknown SBML Fbc Type)") == 0);
}
END_TEST

START_TEST(test_FbcExtension_SBMLtypecode)
{	
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_OBJECTIVE     ,"fbc"), "Objective") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_GENEASSOCIATION,"fbc"), "GeneAssociation") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_ASSOCIATION,"fbc"), "Association") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_FLUXBOUND,"fbc"), "FluxBound") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_FLUXOBJECTIVE,"fbc"), "FluxObjective") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_ASSOCIATION - 1   ,"fbc"), "(Unknown SBML Fbc Type)") == 0);
	fail_unless(strcmp(SBMLTypeCode_toString(SBML_FBC_OBJECTIVE + 1  ,"fbc"), "(Unknown SBML Fbc Type)") == 0);
}
END_TEST

START_TEST(test_FbcExtension_parseGeneAssociation)
{	
  const char* model1 =
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<sbml xmlns:html='http://www.w3.org/1999/xhtml' xmlns='http://www.sbml.org/sbml/level3/version1/core' xmlns:fbc='http://www.sbml.org/sbml/level3/version1/fbc/version1' level='3' version='1' fbc:required='false'>"
    "  <model id='M' name='E' timeUnits='dimensionless'>"
    "    <annotation>"
    "      <listOfGeneAssociations xmlns='http://www.sbml.org/sbml/level3/version1/fbc/version1'>"
    "        <geneAssociation id='ga_1' reaction='R_2DGLCNRx'>"
    "          <gene reference='B3553'/>"
    "        </geneAssociation>"
    "	 </listOfGeneAssociations>"
    "	</annotation>"
    "  </model>"
    "</sbml>"
    ;
  const char* model2 = 
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<sbml xmlns='http://www.sbml.org/sbml/level3/version1/core' xmlns:fbc='http://www.sbml.org/sbml/level3/version1/fbc/version1' metaid='_1f29713d_1639_4193_9935_d6c7ec5255c6' level='3' version='1' fbc:required='false'>"
    " <model metaid='meta' id='id' name='name' timeUnits='dimensionless'>"
    "   <notes>"
    "      <p xmlns='http://www.w3.org/1999/xhtml'>"
    "	   I prevent you from reading annotations ... "
    "	 </p>"
    "   </notes>"
    "  <annotation>"
    "   <listOfGeneAssociations xmlns='http://www.sbml.org/sbml/level3/version1/fbc/version1'>"
    "    <geneAssociation id='ga_1' reaction='MNXR2184_i'>"
    "      <or>"
    "        <gene reference='MCON_1478_I'/>"
    "        <gene reference='MCON_2528_I'/>"
    "        <gene reference='MCON_3321_I'/>"
    "        <gene reference='METACYC_GHHN_1245_MONOMER_I'/>"
    "        <gene reference='METACYC_GHHN_2657_MONOMER_I'/>"
    "        <gene reference='METACYC_GHHN_2722_MONOMER_I'/>"
    "      </or>"
    "    </geneAssociation>"
    "   </listOfGeneAssociations>"
    "  </annotation>"
    " </model>"
    "</sbml>"
    ;

  const char* model3 =
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<sbml xmlns:html='http://www.w3.org/1999/xhtml' xmlns='http://www.sbml.org/sbml/level3/version1/core' xmlns:fbc='http://www.sbml.org/sbml/level3/version1/fbc/version1' level='3' version='1' fbc:required='false'>"
    "  <model id='M' name='E' timeUnits='dimensionless'>"
    "    <annotation>"
    "      <listOfGeneAssociations xmlns='http://www.sbml.org/sbml/level3/version1/fbc/version1'>"
    "        <geneAssociation id='ga_1' reaction='R_2DGLCNRx'>"
    "          <gene reference='B3553.FR1.F121312'/>"
    "        </geneAssociation>"
    "	 </listOfGeneAssociations>"
    "	</annotation>"
    "  </model>"
    "</sbml>"
    ;

  const char* model4 =
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<sbml xmlns:html='http://www.w3.org/1999/xhtml' xmlns='http://www.sbml.org/sbml/level3/version1/core' xmlns:fbc='http://www.sbml.org/sbml/level3/version1/fbc/version1' level='3' version='1' fbc:required='false'>"
    "  <model id='M' name='E' timeUnits='dimensionless'>"
    "    <annotation>"
    "      <listOfGeneAssociations xmlns='http://www.sbml.org/sbml/level3/version1/fbc/version1'>"
    "        <geneAssociation id='ga_1' reaction='R_2DGLCNRx'>"
    "          <gene reference='3.14'/>"
    "        </geneAssociation>"
    "	 </listOfGeneAssociations>"
    "	</annotation>"
    "  </model>"
    "</sbml>"
    ;


  SBMLDocument* doc = readSBMLFromString(model1);
  fail_unless(doc->getModel() != NULL);
  FbcModelPlugin* fbc = dynamic_cast<FbcModelPlugin*>(doc->getModel()->getPlugin("fbc"));
  fail_unless(fbc != NULL);
  fail_unless(fbc->getNumGeneAssociations() == 1);
  SBMLDocument_free(doc);
  doc = readSBMLFromString(model2);
  fail_unless(doc->getModel() != NULL);
  fbc = dynamic_cast<FbcModelPlugin*>(doc->getModel()->getPlugin("fbc"));
  fail_unless(fbc != NULL);
  fail_unless(fbc->getNumGeneAssociations() == 1);
  SBMLDocument_free(doc);
  doc = readSBMLFromString(model3);
  fail_unless(doc->getModel() != NULL);
  fbc = dynamic_cast<FbcModelPlugin*>(doc->getModel()->getPlugin("fbc"));
  fail_unless(fbc != NULL);
  fail_unless(fbc->getNumGeneAssociations() == 1);
  SBMLDocument_free(doc);  
  doc = readSBMLFromString(model4);
  fail_unless(doc->getModel() != NULL);
  fbc = dynamic_cast<FbcModelPlugin*>(doc->getModel()->getPlugin("fbc"));
  fail_unless(fbc != NULL);
  fail_unless(fbc->getNumGeneAssociations() == 1);
  SBMLDocument_free(doc);

  Association* test = Association::parseInfixAssociation("F1.F2.F3 OR F2.f3.f4");
  fail_unless(test != NULL);
  fail_unless(test->toInfix() == "(F1.F2.F3 or F2.f3.f4)");
  test = Association::parseInfixAssociation("3.1 or 3.2");
  fail_unless(test != NULL);
  fail_unless(test->toInfix() == "(3.1 or 3.2)");

}
END_TEST

Suite *
create_suite_FbcExtension (void)
{
  Suite *suite = suite_create("FbcExtension");
  TCase *tcase = tcase_create("FbcExtension");

  tcase_add_checked_fixture(tcase, FbcExtensionTest_setup, FbcExtensionTest_teardown);
 
  tcase_add_test( tcase, test_FbcExtension_getName         );
  tcase_add_test( tcase, test_FbcExtension_getURI          );
  tcase_add_test( tcase, test_FbcExtension_getLevelVersion );
  tcase_add_test( tcase, test_FbcExtension_getSBMLExtensionNamespaces);
  tcase_add_test( tcase, test_FbcExtension_parseGeneAssociation);  
  tcase_add_test( tcase, test_FbcExtension_copy            );
  tcase_add_test( tcase, test_FbcExtension_assignment      );
  tcase_add_test( tcase, test_FbcExtension_clone           );
  tcase_add_test( tcase, test_FbcExtension_registry        );
  tcase_add_test( tcase, test_FbcExtension_typecode        );
  tcase_add_test( tcase, test_FbcExtension_SBMLtypecode    );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND

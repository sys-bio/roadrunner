#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using std::filesystem::path;

using namespace tlp;

class PluginNelderMeadTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginNelderMeadTests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }
};


TEST_F(PluginNelderMeadTests, OPTIMIZE_TEST_MODEL)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    //Plugin* tmplugin = PM->getPlugin("tel_test_model");
    //ASSERT_TRUE(tmplugin != NULL);
    //tmplugin->execute();

    //Plugin* nmplugin = PM->getPlugin("tel_nelder_mead");
    //ASSERT_TRUE(nmplugin != NULL);
    /*
    PropertyBase* sbml = tmplugin->getProperty("Model");
    nmplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());

    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    nmplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    nmplugin->setPropertyValue("InputParameterList", &ipl);
    nmplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    nmplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");
    
    nmplugin->execute();
    //The NM plugin has no status message.
    //EXPECT_EQ(nmplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    PropertyBase* hessian_property = nmplugin->getProperty("Hessian");
    ASSERT_TRUE(hessian_property == NULL);

    PropertyBase* cov_property = nmplugin->getProperty("CovarianceMatrix");
    ASSERT_TRUE(cov_property == NULL);

    PropertyBase* chi_property = nmplugin->getProperty("ChiSquare");
    ASSERT_TRUE(chi_property != NULL);
    double* chisquare = static_cast<double*>(chi_property->getValueHandle());
    EXPECT_NEAR(*chisquare, 76, 70); //Determined empirically.

    PropertyBase* red_chi_property = nmplugin->getProperty("ReducedChiSquare");
    ASSERT_TRUE(red_chi_property != NULL);
    double* reduced_chi = static_cast<double*>(red_chi_property->getValueHandle());
    EXPECT_NEAR(*reduced_chi, 2.8, 2.4); //Determined empirically.

    PropertyBase* outparam_property = nmplugin->getProperty("OutputParameterList");
    ASSERT_TRUE(outparam_property != NULL);
    Properties* outparams = static_cast<Properties*>(outparam_property->getValueHandle());
    PropertyBase* outparam = outparams->getFirst();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "k1");
    double* outparam_val = static_cast<double*>(outparam->getValueHandle());
    EXPECT_NEAR(*outparam_val, 0.925, 0.2);
    EXPECT_TRUE(outparams->getNext() == NULL);

    PropertyBase* conflimit_property = nmplugin->getProperty("ConfidenceLimits");
    ASSERT_TRUE(conflimit_property == NULL);

    PropertyBase* fit_property = nmplugin->getProperty("FittedData");
    ASSERT_TRUE(fit_property != NULL);
    TelluriumData* fit = static_cast<TelluriumData*>(fit_property->getValueHandle());
    EXPECT_EQ(fit->cSize(), 3);
    EXPECT_EQ(fit->rSize(), 14);

    PropertyBase* residuals_property = nmplugin->getProperty("Residuals");
    ASSERT_TRUE(residuals_property != NULL);
    TelluriumData* residuals = static_cast<TelluriumData*>(residuals_property->getValueHandle());
    EXPECT_EQ(residuals->cSize(), 3);
    EXPECT_EQ(residuals->rSize(), 14);

    for (int c = 0; c < fit->cSize(); c++)
    {
        for (int r = 0; r < fit->rSize(); r++)
        {
            double fitval = fit->getDataElement(r, c);
            double origval = exdata->getDataElement(r, c);
            double tol = fmax(abs(origval / 10), 0.0001);
            EXPECT_NEAR(fitval, origval, tol);

            if (c > 0) {
                double residual = abs(residuals->getDataElement(r, c));
                EXPECT_NEAR(abs(origval - fitval), residual, 0.0002);
                EXPECT_LT(residual, 2.2e-5);
                //cout << origval << ", " << fitval << ", " << residual << ", " << abs(origval - fitval) << endl;
            }
        }
    }
    */

    delete PM;
}

TEST_F(PluginNelderMeadTests, OPTIMIZE_NEW_MODEL)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
    /*
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    Plugin* nmplugin = PM->getPlugin("tel_nelder_mead");
    ASSERT_TRUE(nmplugin != NULL);

    string  newModel = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n\
  <model metaid=\"_case00001\" id=\"case00001\" name=\"case00001\">\n\
    <listOfCompartments>\n\
      <compartment id=\"compartment\" name=\"compartment\" size=\"1\" constant=\"true\"/>\n\
    </listOfCompartments>\n\
    <listOfSpecies>\n\
      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialAmount=\"0.00015\"  hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialAmount=\"0\"  hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
      <species id=\"S3\" name=\"S3\" compartment=\"compartment\" initialAmount=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
    </listOfSpecies>\n\
    <listOfParameters>\n\
      <parameter id=\"k1\" name=\"k1\" value=\"1\" constant=\"true\"/>\n\
      <parameter id=\"k2\" name=\"k1\" value=\"0.5\" constant=\"true\"/>\n\
    </listOfParameters>\n\
    <listOfReactions>\n\
      <reaction id=\"reaction1\" name=\"reaction1\" reversible=\"false\">\n\
        <listOfReactants>\n\
          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfReactants>\n\
        <listOfProducts>\n\
          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfProducts>\n\
        <kineticLaw>\n\
          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n\
            <apply>\n\
              <times/>\n\
              <ci> compartment </ci>\n\
              <ci> k1 </ci>\n\
              <ci> S1 </ci>\n\
            </apply>\n\
          </math>\n\
        </kineticLaw>\n\
      </reaction>\n\
      <reaction id=\"reaction2\" reversible=\"false\">\n\
        <listOfReactants>\n\
          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfReactants>\n\
        <listOfProducts>\n\
          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfProducts>\n\
        <kineticLaw>\n\
          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n\
            <apply>\n\
              <times/>\n\
              <ci> compartment </ci>\n\
              <ci> k2 </ci>\n\
              <ci> S2 </ci>\n\
            </apply>\n\
          </math>\n\
        </kineticLaw>\n\
      </reaction>\n\
    </listOfReactions>\n\
  </model>\n\
</sbml>\n\
";

    tmplugin->setPropertyByString("Model", newModel.c_str());
    nmplugin->setPropertyByString("SBML", newModel.c_str());
    
    //tmplugin->setPropertyByString("Seed", "215"); //Will give you nan confidence limits, if allowed.

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Property<double> k2val(0.1, "k2", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    ipl.add(&k2val);
    //tlp::Property tpcre();
    nmplugin->setPropertyValue("InputParameterList", &ipl);
    nmplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2] [S3]");
    nmplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2] [S3]");

    TelluriumData* exdata = NULL;
    unsigned int seed = 10001;
    while (nmplugin->getPropertyValueAsString("StatusMessage") != "converged")
    {
        tmplugin->execute();
        //Set the seed only if the previous iteration fails.
        tmplugin->setPropertyValue("Seed", &seed);
        seed++;

        PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
        exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
        nmplugin->setPropertyValue("ExperimentalData", exdata);

        nmplugin->execute();
    }
    //EXPECT_EQ(nmplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    PropertyBase* hessian_property = nmplugin->getProperty("Hessian");
    ASSERT_TRUE(hessian_property == NULL);

    PropertyBase* cov_property = nmplugin->getProperty("CovarianceMatrix");
    ASSERT_TRUE(cov_property == NULL);

    PropertyBase* chi_property = nmplugin->getProperty("ChiSquare");
    ASSERT_TRUE(chi_property != NULL);
    double* chisquare = static_cast<double*>(chi_property->getValueHandle());
    EXPECT_NEAR(*chisquare, 175, 160); //Determined empirically.

    PropertyBase* red_chi_property = nmplugin->getProperty("ReducedChiSquare");
    ASSERT_TRUE(red_chi_property != NULL);
    double* reduced_chi = static_cast<double*>(red_chi_property->getValueHandle());
    EXPECT_NEAR(*reduced_chi, 4, 3.7); //Determined empirically.

    PropertyBase* outparam_property = nmplugin->getProperty("OutputParameterList");
    ASSERT_TRUE(outparam_property != NULL);
    Properties* outparams = static_cast<Properties*>(outparam_property->getValueHandle());
    PropertyBase* outparam = outparams->getFirst();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "k1");
    double* outparam_val = static_cast<double*>(outparam->getValueHandle());
    EXPECT_NEAR(*outparam_val, 0.98, 0.35);

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "k2");
    outparam_val = static_cast<double*>(outparam->getValueHandle());
    EXPECT_NEAR(*outparam_val, 0.6, 0.17);
    EXPECT_TRUE(outparams->getNext() == NULL);

    PropertyBase* conflimit_property = nmplugin->getProperty("ConfidenceLimits");
    ASSERT_TRUE(conflimit_property == NULL);

    PropertyBase* fit_property = nmplugin->getProperty("FittedData");
    ASSERT_TRUE(fit_property != NULL);
    TelluriumData* fit = static_cast<TelluriumData*>(fit_property->getValueHandle());
    EXPECT_EQ(fit->cSize(), 4);
    EXPECT_EQ(fit->rSize(), 14);

    PropertyBase* residuals_property = nmplugin->getProperty("Residuals");
    ASSERT_TRUE(residuals_property != NULL);
    TelluriumData* residuals = static_cast<TelluriumData*>(residuals_property->getValueHandle());
    EXPECT_EQ(residuals->cSize(), 4);
    EXPECT_EQ(residuals->rSize(), 14);

    for (int c = 0; c < fit->cSize(); c++)
    {
        for (int r = 0; r < fit->rSize(); r++)
        {
            double fitval = fit->getDataElement(r, c);
            double origval = exdata->getDataElement(r, c);
            double tol = fmax(abs(origval / 10), 0.0001);
            EXPECT_NEAR(fitval, origval, tol);

            if (c > 0) {
                double residual = abs(residuals->getDataElement(r, c));
                EXPECT_NEAR(abs(origval - fitval), residual, 0.0002);
                EXPECT_LT(residual, 2.5e-5);
                //cout << origval << ", " << fitval << ", " << residual << ", " << abs(origval - fitval) << endl;
            }
        }
    }
     */

    delete PM;
}

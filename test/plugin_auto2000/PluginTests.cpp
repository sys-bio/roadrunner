#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
//#include "../../wrappers/C/telplugins_properties_api.h"

using std::filesystem::path;

//using namespace testing;
//using namespace std;
using namespace tlp;


class PluginTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginTests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }

    static void SetUpTestSuite() {
        if (PM == NULL)
            PM = new PluginManager(getRoadRunnerPluginBuildDirectory().string());
    }

protected:
    static PluginManager* PM;
};

PluginManager* PluginTests::PM = NULL;


TEST_F(PluginTests, AUTO_2000_Issue_773_no_boundary_species)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "auto2000_2rxn.xml").string().c_str());
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "k");
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("RL0", "0");
    a2kplugin->setPropertyByString("RL1", "5");

    a2kplugin->execute();
}

TEST_F(PluginTests, AUTO_2000_RUN_BIOMOD_203)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "BIOMD0000000203.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "A");
    a2kplugin->setPropertyByString("PCPLowerBound", "10");
    a2kplugin->setPropertyByString("PCPUpperBound", "200");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)          U(2)          U(3)          U(4)          U(5)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 1385);
    EXPECT_EQ(points->at(2), 2189);
    EXPECT_EQ(points->at(3), 3705);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 6);
    EXPECT_EQ(data->rSize(), 3705);
    EXPECT_EQ(data->getColumnNamesAsString(), "A,[OCT4],[SOX2],[NANOG],[Protein],[OCT4_SOX2]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(6, 0), 10.036, 0.001);
    EXPECT_NEAR(data->getDataElement(52, 1), 8.85978, 0.0001);
    EXPECT_NEAR(data->getDataElement(167, 2), 12.3176, 0.0001);
    EXPECT_NEAR(data->getDataElement(623, 3), 0.0469995, 0.0001);
    EXPECT_NEAR(data->getDataElement(1522, 4), 33.0702, 0.0001);
    EXPECT_NEAR(data->getDataElement(2345, 5), 22.6297, 0.0001);
    EXPECT_NEAR(data->getDataElement(3535, 6), 183.378, 0.001);
}

/*
TEST_F(PluginTests, AUTO_2000_RUN_BISTABLE)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "bistable.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Negative");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "k3");
    a2kplugin->setPropertyByString("PCPLowerBound", "0.35");
    a2kplugin->setPropertyByString("PCPUpperBound", "1.5");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 47);
    EXPECT_EQ(points->at(2), 67);
    EXPECT_EQ(points->at(3), 97);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 2);
    EXPECT_EQ(data->rSize(), 97);
    EXPECT_EQ(data->getColumnNamesAsString(), "k3,[x]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(17, 0), 1.16386, 0.0001);
    EXPECT_NEAR(data->getDataElement(93, 1), 2.63297, 0.0001);
}
 */

TEST_F(PluginTests, AUTO_2000_RUN_BISTABLE_IRREVERSIBLE)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "irreversible_bistability.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "Signal");
    a2kplugin->setPropertyByString("PCPLowerBound", "-3");
    a2kplugin->setPropertyByString("PCPUpperBound", "4");
    a2kplugin->setPropertyByString("PreSimulation", "True");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)          U(2)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 116);
    EXPECT_EQ(points->at(2), 255);
    EXPECT_EQ(points->at(3), 361);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 3);
    EXPECT_EQ(data->rSize(), 361);
    EXPECT_EQ(data->getColumnNamesAsString(), "Signal,[R1],[EP]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(17, 0), -2.39214, 0.0001);
    EXPECT_NEAR(data->getDataElement(93, 1), 3.0908, 0.0001);
    EXPECT_NEAR(data->getDataElement(193, 2), 10.5904, 0.0001);
}

TEST_F(PluginTests, TEST_MODEL_STANDARD_RUN)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    EXPECT_TRUE(sbml->getValueAsString().find("<sbml") != string::npos);
    EXPECT_TRUE(sbml->getValueAsString().find("k1") != string::npos);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());
    EXPECT_EQ(sim->cSize(), 3);
    EXPECT_EQ(sim->rSize(), 14);

    double sumdiff = 0;
    for (int r = 0; r < sim->rSize(); r++)
    {
        //The 'time' column should be identical:
        EXPECT_EQ(sim->getDataElement(r, 0), noise->getDataElement(r, 0));

        for (int c = 1; c < sim->cSize(); c++)
        {
            EXPECT_NE(sim->getDataElement(r, c), noise->getDataElement(r, c));
            sumdiff += abs(sim->getDataElement(r, c) - noise->getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff, 3.e-6*28, 1e-4);

    PropertyBase* sig = tmplugin->getProperty("Sigma");
    double* sigma = static_cast<double*>(sig->getValueHandle());
    EXPECT_EQ(*sigma, 3.e-6);

    PropertyBase* seedprop = tmplugin->getProperty("Seed");
    unsigned long* seed = static_cast<unsigned long*>(seedprop->getValueHandle());
    EXPECT_EQ(*seed, 0);
}

TEST_F(PluginTests, TEST_MODEL_CHECK_SEED)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->setPropertyByString("Seed", "1001");
    tmplugin->execute();

    PropertyBase* seedprop = tmplugin->getProperty("Seed");
    unsigned long* seed = static_cast<unsigned long*>(seedprop->getValueHandle());
    EXPECT_EQ(*seed, 1001);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    TelluriumData s1001a(*noise);

    tmplugin->execute();

    TelluriumData s1001b(*noise);

    tmplugin->setPropertyByString("Seed", "1004");
    tmplugin->execute();

    TelluriumData s1004(*noise);

    double sumdiff = 0;
    for (int r = 0; r < s1001a.rSize(); r++)
    {
        //The 'time' column should be identical:
        EXPECT_EQ(s1001a.getDataElement(r, 0), s1001b.getDataElement(r, 0));
        EXPECT_EQ(s1001a.getDataElement(r, 0), s1004.getDataElement(r, 0));

        for (int c = 1; c < s1001a.cSize(); c++)
        {
            EXPECT_EQ(s1001a.getDataElement(r, c), s1001b.getDataElement(r, c));
            sumdiff += abs(s1001a.getDataElement(r, c) - s1004.getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff, 3.e-6 * 2 * 28, 1e-4);
    EXPECT_GT(sumdiff, 0);
}

/*
TEST_F(PluginTests, TEST_MODEL_CHECK_SIGMA)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    TelluriumData siglow(*noise);

    tmplugin->setPropertyByString("Sigma", "10");
    tmplugin->execute();

    TelluriumData sighigh(*noise);

    double sumdiff_low = 0, sumdiff_high = 0;
    for (int r = 0; r < siglow.rSize(); r++)
    {
        for (int c = 1; c < siglow.cSize(); c++)
        {
            sumdiff_low  += abs(siglow.getDataElement(r, c)  - sim->getDataElement(r, c));
            sumdiff_high += abs(sighigh.getDataElement(r, c) - sim->getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff_low, 3.e-6 * 28, 1e-4);
    EXPECT_NEAR(sumdiff_high, 10 * 28, 200);

    PropertyBase* sig = tmplugin->getProperty("Sigma");
    double* sigma = static_cast<double*>(sig->getValueHandle());
    EXPECT_EQ(*sigma, 10);
}

TEST_F(PluginTests, TEST_MODEL_NEW_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

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
    tmplugin->execute();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    EXPECT_EQ(sbml->getValueAsString(), newModel);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 4);
    EXPECT_EQ(noise->rSize(), 14);

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());
    EXPECT_EQ(sim->cSize(), 4);
    EXPECT_EQ(sim->rSize(), 14);
}
 */

TEST_F(PluginTests, LEVENBERG_MARQUARDT_OPTIMIZE_TEST_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    Plugin* lmplugin = PM->getPlugin("tel_levenberg_marquardt");
    ASSERT_TRUE(lmplugin != NULL);

    // reset the value of plugin properties
    lmplugin->resetPropertiesValues();

    Plugin* chiplugin = PM->getPlugin("tel_chisquare");
    ASSERT_TRUE(chiplugin != NULL);

    // reset the value of plugin properties
    chiplugin->resetPropertiesValues();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    lmplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());

    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    lmplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    lmplugin->setPropertyValue("InputParameterList", &ipl);
    lmplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    lmplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    lmplugin->execute();
    EXPECT_EQ(lmplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    PropertyBase* hessian_property = lmplugin->getProperty("Hessian");
    ASSERT_TRUE(hessian_property != NULL);
    TelluriumData* hessian = static_cast<TelluriumData*>(hessian_property->getValueHandle());
    EXPECT_EQ(hessian->rSize(), 1);
    EXPECT_EQ(hessian->cSize(), 1);
    EXPECT_NEAR(hessian->getDataElement(0, 0), 3300, 1500); //Determined empirically.

    PropertyBase* cov_property = lmplugin->getProperty("CovarianceMatrix");
    ASSERT_TRUE(cov_property != NULL);
    TelluriumData* covariance = static_cast<TelluriumData*>(cov_property->getValueHandle());
    EXPECT_EQ(covariance->rSize(), 1);
    EXPECT_EQ(covariance->cSize(), 1);
    EXPECT_NEAR(covariance->getDataElement(0, 0), 0.0003, 0.0001); //Determined empirically.

    PropertyBase* chi_property = lmplugin->getProperty("ChiSquare");
    ASSERT_TRUE(chi_property != NULL);
    double* chisquare = static_cast<double*>(chi_property->getValueHandle());
    EXPECT_NEAR(*chisquare, 76, 70); //Determined empirically.

    PropertyBase* red_chi_property = lmplugin->getProperty("ReducedChiSquare");
    ASSERT_TRUE(red_chi_property != NULL);
    double* reduced_chi = static_cast<double*>(red_chi_property->getValueHandle());
    EXPECT_NEAR(*reduced_chi, 1.0, 0.8); //Determined empirically.

    PropertyBase* outparam_property = lmplugin->getProperty("OutputParameterList");
    ASSERT_TRUE(outparam_property != NULL);
    Properties* outparams = static_cast<Properties*>(outparam_property->getValueHandle());
    PropertyBase* outparam = outparams->getFirst();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "k1");
    double* outparam_val = static_cast<double*>(outparam->getValueHandle());
    EXPECT_NEAR(*outparam_val, 1.0, 0.5);
    EXPECT_TRUE(outparams->getNext() == NULL);

    PropertyBase* conflimit_property = lmplugin->getProperty("ConfidenceLimits");
    ASSERT_TRUE(conflimit_property != NULL);
    Properties* conflimits = static_cast<Properties*>(conflimit_property->getValueHandle());
    PropertyBase* conflimit = conflimits->getFirst();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "k1_confidence");
    double* conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    EXPECT_NEAR(*conflimit_val, 0.03, 0.02);
    EXPECT_TRUE(conflimits->getNext() == NULL);

    PropertyBase* fit_property = lmplugin->getProperty("FittedData");
    ASSERT_TRUE(fit_property != NULL);
    TelluriumData* fit = static_cast<TelluriumData*>(fit_property->getValueHandle());
    EXPECT_EQ(fit->cSize(), 3);
    EXPECT_EQ(fit->rSize(), 14);

    PropertyBase* residuals_property = lmplugin->getProperty("Residuals");
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
                double residual = residuals->getDataElement(r, c);
                EXPECT_NEAR(abs(origval - fitval), residual, 0.0001);
                EXPECT_LT(residual, 0.0001);
                //cout << origval << ", " << fitval << ", " << residual << ", " << abs(origval - fitval) << endl;
            }
        }
    }
}

/*
TEST_F(PluginTests, LEVENBERG_MARQUARDT_OPTIMIZE_HENRICH_WILBERT)
{
    Plugin* lmplugin = PM->getPlugin("tel_levenberg_marquardt");
    ASSERT_TRUE(lmplugin != NULL);

    // reset the value of plugin properties
    lmplugin->resetPropertiesValues();

    Plugin* chiplugin = PM->getPlugin("tel_chisquare");
    ASSERT_TRUE(chiplugin != NULL);

    // reset the value of plugin properties
    chiplugin->resetPropertiesValues();

    lmplugin->setPropertyByString("SBML", (pluginsModelsDir / "HenrichWilbertFit.xml").string().c_str());

    TelluriumData exdata;
    exdata.read((pluginsModelsDir / "wilbertData.dat").string());
    lmplugin->setPropertyValue("ExperimentalData", &exdata);

    Property<double> p0val(6.77, "p0", "", "", "", true);
    Property<double> p1val(1.01, "p1", "", "", "", true);
    Property<double> p4val(1.26, "p4", "", "", "", true);
    Property<double> p6val(35.11, "p6", "", "", "", true);
    Properties ipl;
    ipl.add(&p0val);
    ipl.add(&p1val);
    ipl.add(&p4val);
    ipl.add(&p6val);
    lmplugin->setPropertyValue("InputParameterList", &ipl);
    lmplugin->setPropertyByString("FittedDataSelectionList", "[y1] [y2]");
    lmplugin->setPropertyByString("ExperimentalDataSelectionList", "[y1] [y2]");

    lmplugin->execute();
    EXPECT_EQ(lmplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    PropertyBase* cov_property = lmplugin->getProperty("CovarianceMatrix");
    ASSERT_TRUE(cov_property != NULL);
    TelluriumData* covariance = static_cast<TelluriumData*>(cov_property->getValueHandle());
    EXPECT_EQ(covariance->rSize(), 4);
    EXPECT_EQ(covariance->cSize(), 4);

    //NOTE:  The actual values changed significantly when the simulator changed only
    // very slightly (hstep was changed to next-prev), so I dropped checking the values.
    // They're simply to dependent on very very small integrator changes.
    // If anyone wants to look at the numbers, uncomment and define a 'check_nums' boolean.

    //if (check_nums) {
    //    EXPECT_NEAR(covariance->getDataElement(0, 0), 0.09313539, 0.0000001); //Determined empirically.
    //    EXPECT_NEAR(covariance->getDataElement(1, 3), 1.6250418e-05, 1e-8); //Determined empirically.
    //}
    PropertyBase* hessian_property = lmplugin->getProperty("Hessian");
    ASSERT_TRUE(hessian_property != NULL);
    TelluriumData* hessian = static_cast<TelluriumData*>(hessian_property->getValueHandle());
    EXPECT_EQ(hessian->rSize(), 4);
    EXPECT_EQ(hessian->cSize(), 4);
    //if (check_nums) {
    //    //Spot checks
    //    EXPECT_NEAR(hessian->getDataElement(0, 0), 432.75, 0.01); //Determined empirically.
    //    EXPECT_NEAR(hessian->getDataElement(3, 2), -1023.15, 0.01); //Determined empirically.
    //}

    PropertyBase* chi_property = lmplugin->getProperty("ChiSquare");
    ASSERT_TRUE(chi_property != NULL);
    //if (check_nums) {
    //    double* chisquare = static_cast<double*>(chi_property->getValueHandle());
    //    EXPECT_NEAR(*chisquare, 4.134, 0.001); //Determined empirically.
    //}
    PropertyBase* red_chi_property = lmplugin->getProperty("ReducedChiSquare");
    ASSERT_TRUE(red_chi_property != NULL);
    //if (check_nums) {
    //    double* reduced_chi = static_cast<double*>(red_chi_property->getValueHandle());
    //    EXPECT_NEAR(*reduced_chi, 0.04306, 0.0001); //Determined empirically.
    //}

    PropertyBase* outparam_property = lmplugin->getProperty("OutputParameterList");
    ASSERT_TRUE(outparam_property != NULL);
    Properties* outparams = static_cast<Properties*>(outparam_property->getValueHandle());
    PropertyBase* outparam = outparams->getFirst();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p0");
    double* outparam_val = static_cast<double*>(outparam->getValueHandle());
    //if (check_nums) {
    //    EXPECT_NEAR(*outparam_val, 6.9019, 0.001);
    //}

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p1");
    //if (check_nums) {
    //    outparam_val = static_cast<double*>(outparam->getValueHandle());
    //    EXPECT_NEAR(*outparam_val, 1.01493, 0.0001);
    //}

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p4");
    //if (check_nums) {
    //    outparam_val = static_cast<double*>(outparam->getValueHandle());
    //    EXPECT_NEAR(*outparam_val, 1.09266, 0.0001);
    //}

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p6");
    //if (check_nums) {
    //    outparam_val = static_cast<double*>(outparam->getValueHandle());
    //    EXPECT_NEAR(*outparam_val, 5.04752, 0.0001);
    //}

    EXPECT_TRUE(outparams->getNext() == NULL);


    PropertyBase* conflimit_property = lmplugin->getProperty("ConfidenceLimits");
    ASSERT_TRUE(conflimit_property != NULL);
    Properties* conflimits = static_cast<Properties*>(conflimit_property->getValueHandle());
    PropertyBase* conflimit = conflimits->getFirst();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p0_confidence");
    double* conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    //if (check_nums) {
    //    EXPECT_NEAR(*conflimit_val, 0.124128, 0.0001);
    //}

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p1_confidence");
    //if (check_nums) {
    //    conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    //    EXPECT_NEAR(*conflimit_val, 0.000591761, 0.0001);
    //}

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p4_confidence");
    //if (check_nums) {
    //    conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    //    EXPECT_NEAR(*conflimit_val, 0.192354, 0.0001);
    //}

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p6_confidence");
    //if (check_nums) {
    //    conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    //    EXPECT_NEAR(*conflimit_val, 0.210358, 0.0001);
    //}

    EXPECT_TRUE(conflimits->getNext() == NULL);


    PropertyBase* fit_property = lmplugin->getProperty("FittedData");
    ASSERT_TRUE(fit_property != NULL);
    TelluriumData* fit = static_cast<TelluriumData*>(fit_property->getValueHandle());
    EXPECT_EQ(fit->cSize(), 3);
    EXPECT_EQ(fit->rSize(), 50);

    PropertyBase* residuals_property = lmplugin->getProperty("Residuals");
    ASSERT_TRUE(residuals_property != NULL);
    TelluriumData* residuals = static_cast<TelluriumData*>(residuals_property->getValueHandle());
    EXPECT_EQ(residuals->cSize(), 3);
    EXPECT_EQ(residuals->rSize(), 50);

    for (int c = 0; c < fit->cSize(); c++)
    {
        for (int r = 0; r < fit->rSize(); r++)
        {
            double fitval = fit->getDataElement(r, c);
            double origval = exdata.getDataElement(r, c);
            double tol = fmax(abs(origval / 10), 1.0);
            EXPECT_NEAR(fitval, origval, tol);

            if (c > 0) {
                double residual = abs(residuals->getDataElement(r, c));
                EXPECT_NEAR(abs(origval - fitval), residual, 0.1);
                EXPECT_LT(residual, 1.0);
                //cout << origval << ", " << fitval << ", " << residual << ", " << abs(origval - fitval) << endl;
            }
        }
    }
}
*/

TEST_F(PluginTests, MONTE_CARLO_OPTIMIZE_TEST_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

    // reset the value of plugin properties
    mcplugin->resetPropertiesValues();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    mcplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());
    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    mcplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    mcplugin->setPropertyValue("InputParameterList", &ipl);
    mcplugin->setPropertyByString("NrOfMCRuns", "100");
    mcplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    mcplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    mcplugin->execute();
    //EXPECT_EQ(mcplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    TelluriumData* params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);
    EXPECT_EQ(params->rSize(), 100);
    EXPECT_EQ(params->cSize(), 1);
    for (int r = 0; r < params->rSize(); r++)
    {
        EXPECT_NEAR(params->getDataElement(r, 0), 1.0, 0.2);
    }

    Properties* means = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Means"));
    ASSERT_TRUE(means != NULL);
    Property<double>* mean1 = static_cast<Property<double>*>(means->getFirst());
    EXPECT_EQ(mean1->getName(), "k1");
    EXPECT_NEAR(mean1->getValue(), 1.0, 0.1);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* ci = static_cast<Property<double>* >(conf_intervals->getFirst());
    EXPECT_EQ(ci->getName(), "k1");
    EXPECT_NEAR(ci->getValue(), 0.05, 0.03);

    Properties* percentiles = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Percentiles"));
    ASSERT_TRUE(percentiles != NULL);
    Property<double>* percentile = static_cast<Property<double>*>(percentiles->getFirst());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_25_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.99, 0.12);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_75_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.01, 0.12);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_02.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.96, 0.13);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_97.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.04, 0.13);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_01_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.95, 0.14);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_99_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.05, 0.14);

    EXPECT_TRUE(percentiles->getNext() == NULL);
}

TEST_F(PluginTests, MONTE_CARLO_OPTIMIZE_NELDER_MEAD)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

    // reset the value of plugin properties
    mcplugin->resetPropertiesValues();

    mcplugin->setPropertyByString("MinimizerPlugin", "tel_nelder_mead");

    PropertyBase* sbml = tmplugin->getProperty("Model");
    mcplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());

    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    mcplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    mcplugin->setPropertyValue("InputParameterList", &ipl);
    mcplugin->setPropertyByString("NrOfMCRuns", "100");
    mcplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    mcplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    mcplugin->execute();
    //EXPECT_EQ(mcplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    TelluriumData* params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);
    EXPECT_EQ(params->rSize(), 100);
    EXPECT_EQ(params->cSize(), 1);
    for (int r = 0; r < params->rSize(); r++)
    {
        EXPECT_NEAR(params->getDataElement(r, 0), 1.0, 0.5);
    }

    Properties* means = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Means"));
    ASSERT_TRUE(means != NULL);
    Property<double>* mean1 = static_cast<Property<double>*>(means->getFirst());
    EXPECT_EQ(mean1->getName(), "k1");
    EXPECT_NEAR(mean1->getValue(), 1.0, 0.4);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* ci = static_cast<Property<double>*>(conf_intervals->getFirst());
    EXPECT_EQ(ci->getName(), "k1");
    EXPECT_NEAR(ci->getValue(), 0.2, 0.2);

    Properties* percentiles = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Percentiles"));
    ASSERT_TRUE(percentiles != NULL);
    Property<double>* percentile = static_cast<Property<double>*>(percentiles->getFirst());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_25_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.0, 0.25);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_75_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.03, 0.3);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_02.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.875, 0.32);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_97.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.09, 0.4);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_01_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.875, 0.4);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_99_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.09, 0.4);

    EXPECT_TRUE(percentiles->getNext() == NULL);
}

TEST_F(PluginTests, MONTE_CARLO_CHECK_SEED)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

    // reset the value of plugin properties
    mcplugin->resetPropertiesValues();

    mcplugin->setPropertyByString("Seed", "2001");

    PropertyBase* sbml = tmplugin->getProperty("Model");
    mcplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());

    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    mcplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    mcplugin->setPropertyValue("InputParameterList", &ipl);
    mcplugin->setPropertyByString("NrOfMCRuns", "40");
    mcplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    mcplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    mcplugin->execute();
    //EXPECT_EQ(mcplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    TelluriumData* params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);
    TelluriumData copy(*params);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* cl = static_cast<Property<double>*>(conf_intervals->getFirst());
    ASSERT_TRUE(cl != NULL);
    double cl_one = cl->getValue();

    mcplugin->execute();
    params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);

    EXPECT_EQ(params->rSize(), 40);
    EXPECT_EQ(params->cSize(), 1);
    for (int r = 0; r < params->rSize(); r++)
    {
        EXPECT_EQ(params->getDataElement(r, 0), copy.getDataElement(r, 0));
    }

    conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    cl = static_cast<Property<double>*>(conf_intervals->getFirst());
    ASSERT_TRUE(cl != NULL);

    EXPECT_EQ(cl_one, cl->getValue());
}

TEST_F(PluginTests, NELDER_MEAD_OPTIMIZE_TEST_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    tmplugin->execute();

    Plugin* nmplugin = PM->getPlugin("tel_nelder_mead");
    ASSERT_TRUE(nmplugin != NULL);

    // reset the value of plugin properties
    nmplugin->resetPropertiesValues();

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
}

TEST_F(PluginTests, NELDER_MEAD_OPTIMIZE_NEW_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

    Plugin* nmplugin = PM->getPlugin("tel_nelder_mead");
    ASSERT_TRUE(nmplugin != NULL);

    // reset the value of plugin properties
    nmplugin->resetPropertiesValues();

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
}
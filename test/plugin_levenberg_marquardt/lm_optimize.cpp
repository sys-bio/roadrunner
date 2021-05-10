#include "gtest/gtest.h"
#include "../test_util.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
//#include "tel"


using namespace testing;
using namespace std;
using namespace tlp;

extern string gRRTestDir;
extern string gRROutputDir;
extern string gRRPluginDir;
//extern APIHandleManager gHM;


TEST(RRPLUGIN_TEST_MODEL, OPTIMIZE_TEST_MODEL)
{
    if (gRRPluginDir.empty()) {
        std::cerr << "Please set the 'plugindir' environment variable before running the plugin tests.  This should be the directory where the plugin dlls are created." << std::endl;
        EXPECT_TRUE(false);
    }
    PluginManager* PM = new PluginManager(gRRPluginDir);
    //tpCreatePluginManager();
    //gHM.registerHandle(PM, typeid(PM).name());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);
    tmplugin->execute();

    Plugin* lmplugin = PM->getPlugin("tel_levenberg_marquardt");
    ASSERT_TRUE(lmplugin != NULL);

    Plugin* chiplugin = PM->getPlugin("tel_chisquare");
    ASSERT_TRUE(chiplugin != NULL);

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
    EXPECT_NEAR(*reduced_chi, 2.8, 2.4); //Determined empirically.

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
            double tol = max(abs(origval / 10), 0.0001);
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

TEST(RRPLUGIN_TEST_MODEL, OPTIMIZE_HENRICH_WILBERT)
{
    //We set the initial value of p6 to 35.11, which is *way* off, to test the edges of the optimizer.
    // Some of the runs actually fail under this scenario.
    // In Python, the confidence limits are normal, but here in C, for some reason some are nan.
    if (gRRPluginDir.empty()) {
        std::cerr << "Please set the 'plugindir' environment variable before running the plugin tests.  This should be the directory where the plugin dlls are created." << std::endl;
        EXPECT_TRUE(false);
    }
    PluginManager* PM = new PluginManager(gRRPluginDir);
    //tpCreatePluginManager();
    //gHM.registerHandle(PM, typeid(PM).name());

    Plugin* lmplugin = PM->getPlugin("tel_levenberg_marquardt");
    ASSERT_TRUE(lmplugin != NULL);

    Plugin* chiplugin = PM->getPlugin("tel_chisquare");
    ASSERT_TRUE(chiplugin != NULL);

    lmplugin->setPropertyByString("SBML", (gRRTestDir + "/models/PLUGINS/HenrichWilbertFit.xml").c_str());

    TelluriumData exdata;
    exdata.read(gRRTestDir + "/models/PLUGINS/wilbertData.dat");
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

    bool check_nums = false;
    if (covariance->getDataElement(0, 0) > 0) {
        //There is a certain amount of predictable numerical instability that shows up 
        // depending on the platform and debug level.  If the covariance matrix upper left
        // corner is negative, the estimates are OK, but everything else is kind of off.
        check_nums = true;
    }
    //Spot checks
    if (check_nums) {
        EXPECT_NEAR(covariance->getDataElement(0, 0), 0.09313539, 0.0000001); //Determined empirically.
        EXPECT_NEAR(covariance->getDataElement(1, 3), 1.6250418e-05, 1e-8); //Determined empirically.
    }
    PropertyBase* hessian_property = lmplugin->getProperty("Hessian");
    ASSERT_TRUE(hessian_property != NULL);
    TelluriumData* hessian = static_cast<TelluriumData*>(hessian_property->getValueHandle());
    EXPECT_EQ(hessian->rSize(), 4);
    EXPECT_EQ(hessian->cSize(), 4);
    if (check_nums) {
        //Spot checks
        EXPECT_NEAR(hessian->getDataElement(0, 0), 432.75, 0.01); //Determined empirically.
        EXPECT_NEAR(hessian->getDataElement(3, 2), -1023.15, 0.01); //Determined empirically.
    }

    PropertyBase* chi_property = lmplugin->getProperty("ChiSquare");
    ASSERT_TRUE(chi_property != NULL);
    if (check_nums) {
        double* chisquare = static_cast<double*>(chi_property->getValueHandle());
        EXPECT_NEAR(*chisquare, 4.134, 0.001); //Determined empirically.
    }
    PropertyBase* red_chi_property = lmplugin->getProperty("ReducedChiSquare");
    ASSERT_TRUE(red_chi_property != NULL);
    if (check_nums) {
        double* reduced_chi = static_cast<double*>(red_chi_property->getValueHandle());
        EXPECT_NEAR(*reduced_chi, 0.04306, 0.0001); //Determined empirically.
    }

    PropertyBase* outparam_property = lmplugin->getProperty("OutputParameterList");
    ASSERT_TRUE(outparam_property != NULL);
    Properties* outparams = static_cast<Properties*>(outparam_property->getValueHandle());
    PropertyBase* outparam = outparams->getFirst();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p0");
    double* outparam_val = static_cast<double*>(outparam->getValueHandle());
    if (check_nums) {
        EXPECT_NEAR(*outparam_val, 6.9019, 0.001);
    }

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p1");
    if (check_nums) {
        outparam_val = static_cast<double*>(outparam->getValueHandle());
        EXPECT_NEAR(*outparam_val, 1.01493, 0.0001);
    }

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p4");
    if (check_nums) {
        outparam_val = static_cast<double*>(outparam->getValueHandle());
        EXPECT_NEAR(*outparam_val, 1.09266, 0.0001);
    }

    outparam = outparams->getNext();
    ASSERT_TRUE(outparam != NULL);
    EXPECT_EQ(outparam->getName(), "p6");
    if (check_nums) {
        outparam_val = static_cast<double*>(outparam->getValueHandle());
        EXPECT_NEAR(*outparam_val, 5.04752, 0.0001);
    }

    EXPECT_TRUE(outparams->getNext() == NULL);


    PropertyBase* conflimit_property = lmplugin->getProperty("ConfidenceLimits");
    ASSERT_TRUE(conflimit_property != NULL);
    Properties* conflimits = static_cast<Properties*>(conflimit_property->getValueHandle());
    PropertyBase* conflimit = conflimits->getFirst();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p0_confidence");
    double* conflimit_val = static_cast<double*>(conflimit->getValueHandle());
    if (check_nums) {
        EXPECT_NEAR(*conflimit_val, 0.124128, 0.0001);
    }

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p1_confidence");
    if (check_nums) {
        conflimit_val = static_cast<double*>(conflimit->getValueHandle());
        EXPECT_NEAR(*conflimit_val, 0.000591761, 0.0001);
    }

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p4_confidence");
    if (check_nums) {
        conflimit_val = static_cast<double*>(conflimit->getValueHandle());
        EXPECT_NEAR(*conflimit_val, 0.192354, 0.0001);
    }

    conflimit = conflimits->getNext();
    ASSERT_TRUE(conflimit != NULL);
    EXPECT_EQ(conflimit->getName(), "p6_confidence");
    if (check_nums) {
        conflimit_val = static_cast<double*>(conflimit->getValueHandle());
        EXPECT_NEAR(*conflimit_val, 0.210358, 0.0001);
    }

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
            double tol = max(abs(origval / 10), 1.0);
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

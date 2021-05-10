#pragma hdrstop
#include <sstream>
#include "telLogger.h"
#include "telException.h"
#include "monte_carlo_bs_doc.h"
#include "bsMonteCarlo.h"
#include "telTelluriumData.h"
#include "telUtils.h"
//---------------------------------------------------------------------------

namespace bsmc
{
    rrc::THostInterface* gHostInterface = NULL;
    TELHandle gPluginManager = NULL;
    using namespace std;
    using tlp::StringList;

    MonteCarlo::MonteCarlo()
        :
        CPPPlugin("MonteCarlo-Bootstrap", "Fitting"),

        //Properties.                   //value,                name,                                   hint,                                                           description, alias, readonly);
        mSBML("<none>", "SBML", "SBML document as a string. Model to be used in the fitting"),
        mExperimentalData(TelluriumData(), "ExperimentalData", "Data object holding Experimental data: Provided by client"),
        mInputParameterList(Properties(), "InputParameterList", "List of parameters to fit"),
        mMonteCarloParameters(TelluriumData(), "MonteCarloParameters", "Parameters obtained from a Monte Carlo session."),
        mMeans(Properties(), "Means", "Mean values for each parameter"),
        mConfidenceIntervals(Properties(), "ConfidenceIntervals", "Confidence Intervals for each parameter"),
        mPercentiles(Properties(), "Percentiles", "Percentile values for each parameter"),
        mExperimentalDataSelectionList(StringList(), "ExperimentalDataSelectionList", "Experimental data selection list"),
        mModelDataSelectionList(StringList(), "FittedDataSelectionList", "Model data selection list"),
        mNrOfMCRuns(5, "NrOfMCRuns", "Number of Monte Carlo Data Sets"),
        mMinimizerPlugin("tel_levenberg_marquardt", "MinimizerPlugin", "Which plugin to use as a minimizer.  'tel_levenberg_marquardt' and 'tel_nelder_mead' should both work."),
        mSeed(0, "Seed", "Seed to use for each use of 'execute' (0 to ignore)"),
        mWorker(*this)
    {
        mVersion = "1.0.0";

        //Add plugin properties to property container
        mProperties.add(&mSBML);
        mProperties.add(&mExperimentalData);
        mProperties.add(&mInputParameterList);
        mProperties.add(&mMonteCarloParameters);
        mProperties.add(&mMeans);
        mProperties.add(&mConfidenceIntervals);
        mProperties.add(&mPercentiles);
        mProperties.add(&mExperimentalDataSelectionList);
        mProperties.add(&mModelDataSelectionList);
        mProperties.add(&mNrOfMCRuns);
        mProperties.add(&mMinimizerPlugin);
        mProperties.add(&mSeed);

        //Add the lmfit parameters
        mHint = "Monte Carlo bootstrap algorithm.";
        mDescription = "The Monte Carlo plugin is used to get estimates of a models parameters confidence intervals. This is in a context where experimental data exists and a parameter minimization method, such as Levenberg-Marquardt or Nelder-Mead is used first in order to find a parameter minimum.";

        //The function below assigns property descriptions
        assignPropertyDescriptions();
    }

    MonteCarlo::~MonteCarlo()
    {}

    bool MonteCarlo::isWorking() const
    {
        return mWorker.isRunning();
    }

    unsigned char* MonteCarlo::getManualAsPDF() const
    {
        return pdf_doc;
    }

    size_t MonteCarlo::getPDFManualByteSize()
    {
        return sizeofPDF;
    }

    StringList MonteCarlo::getExperimentalDataSelectionList()
    {
        return mExperimentalDataSelectionList.getValue();
    }

    string MonteCarlo::getStatus()
    {
        stringstream msg;
        msg << Plugin::getStatus();
        return msg.str();
    }

    string MonteCarlo::getImplementationLanguage()
    {
        return bsmc::getImplementationLanguage();
    }

    bool MonteCarlo::resetPlugin()
    {
        if (mWorker.isRunning())
        {
            return false;
        }
        mWorker.reset();
        mTerminate = false;
        mInputParameterList.getValueReference().clear();

        mExperimentalDataSelectionList.getValueReference().clear();
        mModelDataSelectionList.getValueReference().clear();

        //Clear data
        mExperimentalData.clearValue();
        return true;
    }

    string MonteCarlo::getSBML()
    {
        return mSBML.getValue();
    }

    string MonteCarlo::getResult()
    {
        stringstream msg;
        Properties& conf = mConfidenceIntervals;
        if (conf.count())
        {
            msg << "Parameter confidence intervals ========\n";
        }
        else
        {
            msg << "No confidence intervals to report ========";
        }

        for (int i = 0; i < conf.count(); i++)
        {
            Property<double>* confProp = dynamic_cast<Property<double>*> (conf[i]);
            msg << confProp->getName() << " => +/- " << confProp->getValue() << "\n";
        }
        return msg.str();
    }

    bool MonteCarlo::execute(bool inThread)
    {
        try
        {
            RRPLOG(lInfo) << "Executing the MonteCarlo plugin";
            mWorker.start(inThread);
            return true;
        }
        catch (const Exception& ex)
        {
            RRPLOG(lError) << "There was a problem in the execute of the MonteCarlo BootStrap plugin: " << ex.getMessage();
            throw(ex);
        }
        catch (std::exception& e)
        {
            RRPLOG(lError) << "There was a problem in the execute of the MonteCarlo BootStrap plugin: " << e.what();
            throw(e);
        }
    }

    // Plugin factory function
    MonteCarlo* plugins_cc createPlugin()
    {
        //allocate a new object and return it
        return new MonteCarlo();
    }

    const char* plugins_cc getImplementationLanguage()
    {
        return "CPP";
    }

    void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface) {
        gHostInterface = _hostInterface;
    }

    void plugins_cc setPluginManager(TELHandle manager) {
        gPluginManager = manager;
    }

    void MonteCarlo::assignPropertyDescriptions()
    {
        stringstream s;
        s << "The SBML property should be assigned the (XML) \
text that defines the SBML model that is used to fit parameters.";
        mSBML.setDescription(s.str());
        s.str("");

        s << "Experimental data contains the data to be used for fitting input.";
        mExperimentalData.setDescription(s.str());
        s.str("");

        s << "The input parameter list holds the parameters, and their initial values that are to be fitted, e.g. k1, k2. \
The input parameters are properties of the input SBML model";
        mInputParameterList.setDescription(s.str());
        s.str("");

        s << "Parameter values obtained from a Monte Carlo session. The data is ordered so each set of values for parameter 1 is in column 1, and parameters 2 values are in column 2, etc.";
        mMonteCarloParameters.setDescription(s.str());
        s.str("");

        s << "The mean values parameter list holds resulting mean values of the obtained Monte Carlo Parameters";
        mMeans.setDescription(s.str());
        s.str("");

         s << "The confidence intervals list holds the 95% confidence intervals, as calculated using the obtained Monte Carlo Parameters";
        mConfidenceIntervals.setDescription(s.str());
        s.str("");

        s << "The percentiles list holds the calculable percentiles from the obtained Monte Carlo Parameters.  If at least four repeats are made, the 25 and 75 percentiles are added; at 40 repeats, the 2.5 and 97.5 percentiles are added; at 100 repeats, the 1 and 99 percentiles are added.";
        mPercentiles.setDescription(s.str());
        s.str("");

        s << "The data input may contain multiple columns of data. The Experimental data selection list \
should contain the columns in the input data that is intended to be used in the fitting.";
        mExperimentalDataSelectionList.setDescription(s.str());
        s.str("");

        s << "The model data selection list contains the selections for which model data will be genereated.  \
Internally this data is used to calcualte residual data.";
        mModelDataSelectionList.setDescription(s.str());
        s.str("");

        s << "The number of Monte Carlo runs, i.e. the number of Monte Carlo data sets to generate.";
        mNrOfMCRuns.setDescription(s.str());
        s.str("");

        s << "The Monte Carlo plugin will use a minimization routine internally. The routine to use can be set using the MinimizerPlugin property.";
        mMinimizerPlugin.setDescription(s.str());
        s.str("");

    }

}
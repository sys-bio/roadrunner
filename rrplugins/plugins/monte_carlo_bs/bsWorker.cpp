#pragma hdrstop
#include "bsWorker.h"
#include <time.h>
#include <algorithm>
#include "telLogger.h"
#include "rrRoadRunnerOptions.h"
#include "telException.h"
#include "telTelluriumData.h"
#include "bsMonteCarlo.h"
#include "telUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
#include "telPluginManager.h"
#include "telplugins_c_api.h"
//---------------------------------------------------------------------------

namespace bsmc
{
    extern rrc::THostInterface* gHostInterface;
    extern tlpc::TELHandle gPluginManager;
    using namespace std;
    using namespace tlpc;

    double getRandomElement(const vector<double>& vec);

    bsWorker::bsWorker(MonteCarlo& host)
        :
        mParent(host),
        mRandom(new tlp::Random((unsigned long)time(NULL)))
    {
    }

    bsWorker::~bsWorker()
    {
        delete mRandom;
    }

    bool bsWorker::setup()
    {
        mMinimizerPlugin = (Plugin*)gHostInterface->getPlugin(gPluginManager, mParent.mMinimizerPlugin.getValue().c_str());

        if (!mMinimizerPlugin)
        {
            stringstream msg;
            msg << "Failed loading plugin " << mParent.mMinimizerPlugin.getValue() << ". ";
            msg << "Monte Carlo Plugin cannot proceed.  This value is set with the MinimizerPlugin property,";
            msg << " and should probably be either 'tel_levenberg_marquardt' or 'tel_nelder_mead'.  If it";
            msg << " already is one of those two values, those plugin dlls may not have been installed.";
            throw(Exception(msg.str()));
        }
        reset();
        return true;
    }

    void bsWorker::reset()
    {
        mResiduals.clear();
        for (int i = 0; i < mMCDataSets.size(); i++)
        {
            delete mMCDataSets[i];
        }
        mMCDataSets.clear();

        for (int i = 0; i < mMCParameters.size(); i++)
        {
            mMCParameters[i].clear();
        }
        mMCParameters.clear();
    }

    void bsWorker::start(bool runInThread)
    {
        unsigned long seed = *(unsigned long*)(mParent.getPropertyValueHandle("Seed"));
        if (seed != 0.0)
        {
            delete mRandom;
            mRandom = new tlp::Random(seed);
        }
        if (runInThread)
        {
            if (mThread.isRunning())
            {
                RRPLOG(lError) << "Tried to start an already working thread!";
                return;
            }

            mThread.start(*this);    //Will spawn the run function below in a thread
        }
        else
        {
            run();
        }
    }

    void bsWorker::run()
    {
        workerStarted();

        //The user may have aborted the monte carlo run. check if so here..
        if (mParent.mTerminate)
        {
            //user did set the terminate flag to true.. discard any minimization data and get out of the
            //plugin execute code..
            RRPLOG(lInfo) << "The Monte Carlo plugin was terminated. Aborting";
            workerFinished();
            return;
        }

        //Get to work..
        if (!setup())
        {
            RRPLOG(lError) << "Failed setting up the Monte Carlo Bootstrap plugin. Aborting.";
            return;
        }

        workerProgress();
        //Create MC data sets
        //First get initial residuals
        if (!createInitialResiduals())
        {
            RRPLOG(lError) << "Failed creating initial residuals in Monte Carlo plugin.";
            return;
        }
        else
        {
            RRPLOG(lDebug) << "Monte Carlo initial residuals created.";
        }

        if (!createMonteCarloDataSets())
        {
            RRPLOG(lError) << "Failed creating Monte Carlo Data sets.";
            return;
        }
        else
        {
            RRPLOG(lDebug) << "Monte Carlo Data sets was created.";
        }

        //Now fit each data set and collect parameter statistics
        //Each fit should be carried out in a thread.. Skip the threading for now
        for (int i = 0; i < mParent.mNrOfMCRuns; i++)
        {

            Properties parameters = getParameters(mMCDataSets[i]);
            mParent.mCurrentParameters.setValue(parameters);
            mMCParameters.push_back(parameters);
            workerProgress();
        }

        TelluriumData& parasData = mParent.mMonteCarloParameters.getValueReference();
        Properties& fitParas = mParent.mInputParameterList.getValueReference();
        parasData.reSize(mParent.mNrOfMCRuns, mParent.mInputParameterList.getValue().count());

        //Setup column header
        StringList hdr;
        for (int col = 0; col < fitParas.count(); col++)
        {
            hdr.add(fitParas[col]->getName());
        }

        parasData.setColumnNames(hdr);

        //Copy parameters to parameters container
        for (int i = 0; i < mParent.mNrOfMCRuns; i++)
        {
            RRPLOG(lInfo) << "MC Run: " << i;
            Properties& vecParas = mMCParameters[i];
            for (int para = 0; para < vecParas.count(); para++)
            {
                double value = *((double*)vecParas[para]->getValueHandle());
                RRPLOG(lInfo) << vecParas[para]->getName() << " = " << value;
                parasData(i, para) = value;
            }
        }

        //Calculate confidence intervals
        Properties& inpParaList = mParent.mInputParameterList.getValueReference();
        Properties& means = mParent.mMeans.getValueReference();
        means.clear();
        Properties& confidenceIntervals = mParent.mConfidenceIntervals.getValueReference();
        confidenceIntervals.clear();
        Properties& percentiles = mParent.mPercentiles.getValueReference();
        percentiles.clear();

        for (int para = 0; para < inpParaList.count(); para++)
        {
            vector<double> values;
            for (int i = 0; i < mParent.mNrOfMCRuns && i < mMCParameters.size(); i++)
            {
                Properties& paras = mMCParameters[i];
                double val = *(double*)paras[para]->getValueHandle();
                values.push_back(val);
            }
            //Do the statistics for each parameter
            if (values.size() > 0) {
                double mean = getMean(values);
                Property<double>* prop = new Property<double>(mean, inpParaList[para]->getName());
                means.add(prop);
                RRPLOG(lInfo) << inpParaList[para]->getName() << "mean: " << mean << endl;
                sort(values.begin(), values.end());
                if (values.size() >= 4) {
                    //Calculate the quartiles
                    size_t index = roundl(values.size() / 4.0);
                    double lowest = values[index];
                    double highest = values[values.size() - index - 1];

                    prop = new Property<double>(lowest, inpParaList[para]->getName() + "_25_percentile");
                    percentiles.add(prop);
                    prop = new Property<double>(highest, inpParaList[para]->getName() + "_75_percentile");
                    percentiles.add(prop);
                }
                if (values.size() >= 40) {
                    size_t index = roundl(values.size() / 40.0);
                    double lowest = values[index];
                    double highest = values[values.size() - index - 1];
                    double interval = (highest - lowest) / 2;

                    prop = new Property<double>(interval, inpParaList[para]->getName());
                    confidenceIntervals.add(prop);

                    prop = new Property<double>(lowest, inpParaList[para]->getName() + "_02.5_percentile");
                    percentiles.add(prop);
                    prop = new Property<double>(highest, inpParaList[para]->getName() + "_97.5_percentile");
                    percentiles.add(prop);

                    RRPLOG(lInfo) << inpParaList[para]->getName() << "95% confidence interval: " << lowest << " - " << highest << endl;
                }
                if (values.size() >= 100) {
                    size_t index = roundl(values.size() / 100.0);
                    double lowest = values[index];
                    double highest = values[values.size() - index - 1];

                    prop = new Property<double>(lowest, inpParaList[para]->getName() + "_01_percentile");
                    percentiles.add(prop);
                    prop = new Property<double>(highest, inpParaList[para]->getName() + "_99_percentile");
                    percentiles.add(prop);
                }
            }
        }
        workerFinished();
    }

    Properties bsWorker::getParameters(TelluriumData* mcData)
    {
        mMinimizerPlugin->resetPlugin();

        //Add input parameters, only the checked ones
        Properties* inputParameters = (Properties*)mParent.mInputParameterList.getValueHandle();
        mMinimizerPlugin->setPropertyValue("InputParameterList", inputParameters);

        //Set input data to fit to
        mMinimizerPlugin->setPropertyValue("ExperimentalData", mcData);

        //Add species to minimization data structure.. The species are defined in the input data
        StringList modelDataSelectionList = mParent.mModelDataSelectionList.getValue();
        mMinimizerPlugin->setPropertyValue("FittedDataSelectionList", &modelDataSelectionList);

        StringList expDataSelectionList = mParent.mExperimentalDataSelectionList.getValue();
        mMinimizerPlugin->setPropertyValue("ExperimentalDataSelectionList", &expDataSelectionList);

        //Requirement => the modelDataSelection list must be equal or larger than the expSelectionlist
        if (expDataSelectionList.count() > modelDataSelectionList.count())
        {
            RRPLOG(lError) << "The minimization engine requires the model selection list to be equal or larger than Experimental selection list";
            RRPLOG(lError) << "Exiting Monte Carlo.";
            return false;
        }

        string strVal = mParent.mSBML.getValue();
        mMinimizerPlugin->setPropertyValue("SBML", &strVal);

        mMinimizerPlugin->execute();

        //Check on success of fitting. If failing, bail
        //Extract the parameters
        Properties* newProps = (Properties*)mMinimizerPlugin->getPropertyValueHandle("OutputParameterList");
        if (!newProps)
        {
            throw(Exception("Failed to get 'OutputParameterList' of the minimizer plugin from Monte Carlo"));
        }

        RRPLOG(lDebug) << "Properties: " << (*newProps);
        return *newProps;
    }

    bool bsWorker::createInitialResiduals()
    {
        //Use the current minimization plugin to run one minimization and then use
        //the result to create residuals
        //Reset on each run
        mMinimizerPlugin->resetPlugin();

        //Add input parameters, only the checked ones
        Properties* inputParameters = (Properties*)mParent.mInputParameterList.getValueHandle();
        mMinimizerPlugin->setPropertyValue("InputParameterList", inputParameters);

        //Set input data to fit to
        TelluriumData* data = (TelluriumData*)mParent.mExperimentalData.getValueHandle();
        mMinimizerPlugin->setPropertyValue("ExperimentalData", data);

        //Add species to minimization data structure.. The species are defined in the input data
        StringList modelDataSelectionList = mParent.mModelDataSelectionList.getValue();
        mMinimizerPlugin->setPropertyValue("FittedDataSelectionList", &modelDataSelectionList);

        StringList expDataSelectionList = mParent.mExperimentalDataSelectionList.getValue();
        mMinimizerPlugin->setPropertyValue("ExperimentalDataSelectionList", &expDataSelectionList);

        //Requirement => the modelDataSelection list must be equal or larger than the expSelectionlist
        if (expDataSelectionList.count() > modelDataSelectionList.count())
        {
            RRPLOG(lError) << "The minimization engine requires the model selection list to be equal or larger than Experimental selection list";
            RRPLOG(lError) << "Exiting Monte Carlo.";
            return false;
        }

        string strVal = mParent.mSBML.getValue();
        mMinimizerPlugin->setPropertyValue("SBML", &strVal);

        mMinimizerPlugin->execute();

        //Check on success of fitting. If failing, bail the monte carlo..

        //We should now have residuals
        TelluriumData* residuals = (TelluriumData*)mMinimizerPlugin->getPropertyValueHandle("Residuals");

        RRPLOG(lDebug) << "Logging residuals: ";
        RRPLOG(lDebug) << *(residuals);

        //Add residuals to double vectgor
        for (int col = residuals->isFirstColumnTime() ? 1 : 0; col < residuals->cSize(); col++)
        {
            for (int row = 0; row < residuals->rSize(); row++)
            {
                mResiduals.push_back(residuals->getDataElement(row, col));
            }
        }

        return true;
    }
    bool bsWorker::createMonteCarloDataSets()
    {
        TelluriumData& expData = mParent.mExperimentalData;
        TelluriumData* initialFit = (TelluriumData*)mMinimizerPlugin->getPropertyValueHandle("FittedData");

        //Create data sets, use fitted data as the "base", to add residuals to later on
        for (int i = 0; i < mParent.mNrOfMCRuns; i++)
        {
            mMCDataSets.push_back(new TelluriumData((*initialFit)));

        }

        TelluriumData all;
        //Create MC data by randomly adding residuals from the residuals vector to each point in each data set

        for (int i = 0; i < mMCDataSets.size(); i++)
        {
            TelluriumData& data = *(mMCDataSets[i]);
            int startCol = data.isFirstColumnTime() ? 1 : 0;
            for (int col = startCol; col < data.cSize(); col++)
            {
                for (int row = 0; row < data.rSize(); row++)
                {
                    data(row, col) = data(row, col) + getRandomElement(mResiduals, *mRandom);
                }
            }
            all.append(data);
        }

        all.write("MCDataSets.dat");
        return true;
    }

    void bsWorker::workerStarted()
    {
        mParent.mIsWorking = true;
        if (mParent.mWorkStartedEvent)
        {
            mParent.mWorkStartedEvent(mParent.mWorkStartedData1, mParent.mWorkStartedData2);
        }
    }

    void bsWorker::workerProgress()
    {
        if (mParent.mWorkProgressEvent)
        {
            mParent.mWorkProgressEvent(mParent.mWorkProgressData1, mParent.mWorkProgressData2);
        }
    }

    void bsWorker::workerFinished()
    {
        mParent.mIsWorking = false;//Set this flag before event so client can query plugin about termination
        if (mParent.mWorkFinishedEvent)
        {
            mParent.mWorkFinishedEvent(mParent.mWorkFinishedData1, mParent.mWorkFinishedData2);
        }
    }

    bool bsWorker::isRunning() const
    {
        return mThread.isRunning();
    }

}
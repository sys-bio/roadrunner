#pragma hdrstop
#include "nmWorker.h"
#include "rrRoadRunnerOptions.h"
#include "rr-libstruct/lsLA.h"
#include "lib/nmsimplex.h"
#include "telLogger.h"
#include "telException.h"
#include "telTelluriumData.h"
#include "telUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
#include "telPluginManager.h"
#include "nmNelderMead.h"
#include "nmObjectiveFunction.h"
//---------------------------------------------------------------------------

namespace nmfit
{
    extern rrc::THostInterface* gHostInterface;
    extern tlpc::TELHandle gPluginManager;

    using namespace rrc;
    using namespace std;
    using namespace tlp;

    nmWorker::nmWorker(NelderMead& host)
        :
        mTheHost(host)
    {}

    bool nmWorker::isRunning() const
    {
        return mThread.isRunning();
    }

    int nmWorker::getNumberOfParameters()
    {
        Properties& inParas = mTheHost.mInputParameterList.getValueReference();
        return inParas.count();
    }

    void nmWorker::start(bool runInThread)
    {
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

    void nmWorker::run()
    {
        workerStarted();
        setupRoadRunner();

        StringList& species = mTheHost.mExperimentalDataSelectionList.getValueReference();
        RRPLOG(lInfo) << "The following species are selected: " << species.asString();

        Properties& inParas = mTheHost.mInputParameterList.getValueReference();
        Properties& outParas = mTheHost.mOutputParameterList.getValueReference();
        int         nrOfParameters = inParas.count();

        RRPLOG(lInfo) << "The following parameters are to be minimized";
        for (int i = 0; i < nrOfParameters; i++)
        {
            RRPLOG(lInfo) << inParas[i]->getName() << " with initial value: " << inParas[i]->getValueAsString();
        }

        mTheHost.mNrOfIter.setValue(0);
        mTheHost.mNrOfFuncIter.setValue(0);
        mTheHost.mNorm.setValue(0.0);
        mTheHost.mStatusMessage.setValueFromString("failed");

        //Parameters for the Algorithm..
        double* intialParameters = new double[nrOfParameters];

        for (int i = 0; i < nrOfParameters; i++)
        {
            intialParameters[i] = *(double*)inParas[i]->getValueHandle();
        }

        //Execute minimizer
        double min = simplex2(NelderMeadObjectiveFunction,
            &mTheHost,
            intialParameters,
            nrOfParameters,
            mTheHost.mEpsilon,
            mTheHost.mScale,
            NULL,
            mTheHost.mMaxIterations,
            mTheHost.mALPHA,
            mTheHost.mBETA,
            mTheHost.mGAMMA
        );

        //Populate with data to report back
        for (int i = 0; i < nrOfParameters; ++i)
        {
            outParas.add(new Property<double>(intialParameters[i], inParas[i]->getName(), ""), true);
        }

        //The user may have aborted the minimzation... check here..
        if (mTheHost.mTerminate)
        {
            //user did set the terminate flag to true.. discard any minimization data and get out of the
            //plugin execute code..
            RRPLOG(lInfo) << "The Nelder-Mead minimization was terminated.. aborting";
            workerFinished();
            return;
        }

        //Post fitting data calculations
        RRPLOG(lDebug) << "==================== Fitting Result ================================";
        RRPLOG(lDebug) << "Nr of function evaluations: " << mTheHost.mNrOfIter.getValue();
        RRPLOG(lDebug) << "Minimized parameter values: ";
        for (int i = 0; i < inParas.count(); ++i)
        {
            RRPLOG(lDebug) << "\t" << outParas[i]->getName() << " = " << outParas[i]->getValueAsString();
        }

        RRPLOG(lDebug) << "Final Norm:  " << mTheHost.mNorm.getValue();
        postFittingWork();
        workerFinished();
    }

    bool nmWorker::setupRoadRunner()
    {

        if (mTheHost.rrHandle)
        {
            delete mTheHost.rrHandle;
        }

        mTheHost.rrHandle = gHostInterface->createRRInstance();
        bool ret = gHostInterface->loadSBML(mTheHost.rrHandle, mTheHost.mSBML.getValue().c_str());
        if (!ret)
        {
            string msg = "Failed to load SBML model: ";
            msg += gHostInterface->getLastError();
            throw Exception(msg);
        }
        gHostInterface->setTimeCourseSelectionList(mTheHost.rrHandle, mTheHost.getExperimentalDataSelectionList().asString().c_str());

        return true;
    }

    void nmWorker::postFittingWork()
    {
        Properties* inParas = (Properties*)mTheHost.mInputParameterList.getValueHandle();
        int nrOfParameters = inParas->count();

        //Create model and residuals data
        createModelData(mTheHost.mModelData.getValuePointer());
        createResidualsData(mTheHost.mResidualsData.getValuePointer());

        //Copy Norms
        mTheHost.rNormsData.reSize(mTheHost.mTheNorms.size(), 1);
        mTheHost.rNormsData.setColumnNames(StringList("Norm"));
        for (int r = 0; r < mTheHost.mTheNorms.size(); r++)
        {
            mTheHost.rNormsData(r, 0) = mTheHost.mTheNorms[r];
        }

        //Calculate standardized residuals
        TelluriumData& residuals = *(TelluriumData*)mTheHost.mResidualsData.getValueHandle();


        //When there is a bad fit, residuals get really large and so
        //several of the statistics below overflow

        //Populate the standardized residuals
        try
        {
            TelluriumData& stdRes = *(TelluriumData*)mTheHost.mStandardizedResiduals.getValueHandle();
            stdRes = getStandardizedPopulations(residuals);

            //Create a probability plot for the residuals
            TelluriumData& probPlot = *(TelluriumData*)mTheHost.mNormalProbabilityOfResiduals.getValueHandle();
            probPlot = getNormalProbabilityPlot(stdRes);

            calculateChiSquare();
            //calculateHessian();
            //calculateCovariance();
            //calculateConfidenceLimits();
        }
        catch (...)
        {
            RRPLOG(lError) << "There was problems calculating fit statistics";
        }

    }

    void nmWorker::calculateChiSquare()
    {
        Properties* inParas = (Properties*)mTheHost.mInputParameterList.getValueHandle();
        int nrOfParameters = inParas->count();

        //Calculate ChiSquare(s)
        TelluriumData& modelData = *(TelluriumData*)mTheHost.mModelData.getValuePointer();
        TelluriumData& obsData = *(TelluriumData*)mTheHost.mExperimentalData.getValuePointer();

        Plugin* chi = mTheHost.getChiSquarePlugin();
        
        Property<TelluriumData>* para = dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
        para->setValue(obsData);

        para = dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
        para->setValue(modelData);

        Property<int>* intPara = dynamic_cast<Property<int>*>(chi->getProperty("NrOfModelParameters"));
        intPara->setValue(nrOfParameters);

        //Calculate Chi square
        chi->execute();

        Property<double>* chiSquare = dynamic_cast<Property<double>*>(chi->getProperty("ChiSquare"));
        Property<double>* rChiSquare = dynamic_cast<Property<double>*>(chi->getProperty("ReducedChiSquare"));

        mTheHost.mChiSquare.setValue(chiSquare->getValue());
        mTheHost.mReducedChiSquare.setValue(rChiSquare->getValue());

        RRPLOG(lDebug) << "Chi Square = " << chiSquare->getValue();
        RRPLOG(lDebug) << "Reduced Chi Square = " << rChiSquare->getValue();
    }

    double nmWorker::getChi(const Properties& parameters)
    {
        RRPLOG(lDebug) << "Getting chisquare using parameters: " << parameters;
        //Reset RoadRunner
        gHostInterface->reset(mTheHost.rrHandle);

        for (int i = 0; i < parameters.count(); i++)
        {
            Property<double>* para = (Property<double>*) (parameters[i]);
            gHostInterface->setValue(mTheHost.rrHandle, para->getName().c_str(), para->getValue());
        }

        TelluriumData* expData = (TelluriumData*)mTheHost.mExperimentalData.getValueHandle();

        gHostInterface->reset(mTheHost.rrHandle);

        bool ret = gHostInterface->simulateExNoReturn(mTheHost.rrHandle, expData->getTimeStart(), expData->getTimeEnd(), expData->rSize());
        if (!ret)
        {
            string msg = "Roadrunner simulaton failed when calculating chi in the Nelder-Mead plugin.";
            RRPLOG(lError) << msg;
            //throw std::runtime_error(msg.c_str());
        }
        DoubleMatrix* modelData = (DoubleMatrix*)gHostInterface->getSimulationResultAsDoubleMatrix(mTheHost.rrHandle);

        TelluriumData& obsData = *(TelluriumData*)mTheHost.mExperimentalData.getValuePointer();
        Plugin* chi = mTheHost.getChiSquarePlugin();

        Property<TelluriumData>* para = dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
        para->setValue(obsData);

        para = dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
        TelluriumData td;
        td.setData(*modelData);
        para->setValue(td);

        Property<int>* intPara = dynamic_cast<Property<int>*>(chi->getProperty("NrOfModelParameters"));
        intPara->setValue(getNumberOfParameters());

        //Calculate Chi square
        chi->execute();
        Property<double>* chiSquare = dynamic_cast<Property<double>*>(chi->getProperty("ChiSquare"));
        return chiSquare->getValue();
    }

    void nmWorker::calculateHessian()
    {
        double eta = 6.06e-6;
        int nrOfParameters = mTheHost.mOutputParameterList.getValueReference().count();

        Properties& ref = mTheHost.mOutputParameterList.getValueReference();

        Properties copy; //Gotta add a copy/assignemnt opertor to Properties..
        for (int i = 0; i < ref.count(); i++)
        {
            copy.add(ref[i]);
        }

        //We need to calculate ChiSquares for various perturbatin around a parameter value, in order to get the Hessian
        TelluriumData mat(nrOfParameters, nrOfParameters);

        for (int i = 0; i < nrOfParameters; i++)
        {
            for (int j = 0; j < nrOfParameters; j++)
            {
                if (i == j)//Diagonal elements
                {
                    Property<double>* p = (Property<double>*) copy[i];
                    double origValue = p->getValue();
                    double hi = eta * fabs(origValue);

                    p->setValue(origValue + hi);
                    double t1 = getChi(copy);

                    p->setValue(origValue - hi);
                    double t2 = getChi(copy);

                    p->setValue(origValue);
                    double t3 = getChi(copy);
                    mat(i, j) = (t1 + t2 - 2.0 * t3) / (hi * hi);
                }
                else //Off diagonal elements
                {
                    Property<double>* pi = (Property<double>*) copy[i];
                    double origValueI = pi->getValue();

                    Property<double>* pj = (Property<double>*) copy[j];
                    double origValueJ = pj->getValue();
                    double hi = eta * fabs(origValueI);
                    double hj = eta * fabs(origValueJ);

                    //Term1
                    pi->setValue(origValueI + hi);
                    pj->setValue(origValueJ + hj);
                    double t1 = getChi(copy);

                    //Term2
                    pj->setValue(origValueJ);
                    double t2 = getChi(copy);

                    //Term3
                    pi->setValue(origValueI);
                    pj->setValue(origValueJ + hj);
                    double t3 = getChi(copy);

                    //Term4
                    pi->setValue(origValueI);
                    pj->setValue(origValueJ);
                    double t4 = getChi(copy);
                    mat(i, j) = (t1 - t2 - t3 + t4) / (hi * hj);

                }
                //Reset parameter values

            }
        }
        mTheHost.mHessian.setValue(mat);
    }

    void nmWorker::calculateCovariance()
    {
        //Check if Hessain is invertible
        DoubleMatrix mat = mTheHost.mHessian.getValue().getData();

        ls::ComplexMatrix temp(mat); //Get a complex matrix from a double one. Imag part is zero

        ls::ComplexMatrix Inv = GetInverse(temp);

        TelluriumData temp2(mat.RSize(), mat.CSize());
        temp2.setData(Inv);

        mTheHost.mCovarianceMatrix.setValue(temp2);
    }


    void nmWorker::workerStarted()
    {
        mTheHost.mIsWorking = true;
        if (mTheHost.mWorkStartedEvent)
        {
            mTheHost.mWorkStartedEvent(mTheHost.mWorkStartedData1, mTheHost.mWorkStartedData2);
        }
    }

    void nmWorker::workerFinished()
    {
        mTheHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
        if (mTheHost.mWorkFinishedEvent)
        {
            mTheHost.mWorkFinishedEvent(mTheHost.mWorkFinishedData1, mTheHost.mWorkFinishedData2);
        }
    }

    void nmWorker::createModelData(TelluriumData* _data)
    {
        TelluriumData& data = *(_data);
        TelluriumData* expData = (TelluriumData*)mTheHost.mExperimentalData.getValueHandle();

        Properties& outParas = *(Properties*)mTheHost.mOutputParameterList.getValueHandle();
        int nrOfParameters = outParas.count();

        //We now have the parameters
        StringList selList("time");
        selList.append(mTheHost.mModelDataSelectionList.getValue());

        data.reSize(expData->rSize(), selList.count());
        gHostInterface->reset(mTheHost.rrHandle);
        gHostInterface->setTimeCourseSelectionList(mTheHost.rrHandle, selList.asString().c_str());

        for (int i = 0; i < nrOfParameters; i++)
        {
            Property<double>* para = (Property<double>*) (outParas[i]);
            gHostInterface->setValue(mTheHost.rrHandle, para->getName().c_str(), para->getValue());
        }

        rr::SimulateOptions options;
        options.start = expData->getTimeStart();
        options.duration = expData->getTimeEnd() - expData->getTimeStart();
        options.steps = expData->rSize() - 1;

        gHostInterface->reset(mTheHost.rrHandle);

        bool ret = gHostInterface->simulateExNoReturn(mTheHost.rrHandle, expData->getTimeStart(), expData->getTimeEnd(), expData->rSize());
        if (!ret)
        {
            string msg = "Roadrunner simulaton failed when creating model data in the Nelder-Mead plugin.";
            RRPLOG(lError) << msg;
            throw std::runtime_error(msg.c_str());
        }
        DoubleMatrix* modelData = (DoubleMatrix*)gHostInterface->getSimulationResultAsDoubleMatrix(mTheHost.rrHandle);
        if (modelData)
        {
            data.setData(*modelData);
        }
    }

    void nmWorker::createResidualsData(TelluriumData* _data)
    {
        TelluriumData& resData = *(_data);
        //We now have the parameters
        TelluriumData& obsData = (mTheHost.mExperimentalData.getValueReference());
        TelluriumData& modData = (mTheHost.mModelData.getValueReference());

        resData.reSize(modData.rSize(), modData.cSize());

        //setup coulumn names
        resData.setColumnNames(modData.getColumnNames());
        StringList& species = mTheHost.mExperimentalDataSelectionList.getValueReference();

        for (int sel = 0; sel < species.count() + 1; sel++)    //selection 1 becuase of time column..
        {
            for (int i = 0; i < obsData.rSize(); i++)
            {
                if (sel == 0)
                {
                    resData(i, sel) = modData(i, sel);    //Time
                }
                else
                {
                    //The modData may contain data for other species than that was fitted..
                    //We need to find out what coulmn correspond to what..
                    string specie = obsData.getColumnName(sel);
                    int colNr = modData.getColumnIndex(specie);
                    if (colNr != -1)
                    {
                        resData(i, sel) = obsData(i, sel) - modData(i, colNr);
                    }
                    else
                    {
                        RRPLOG(lError) << "Unable to find species '" << specie << "' in the loaded model, but it is one of the expected outputs.";
                    }
                }
            }
        }
    }

    void nmWorker::calculateConfidenceLimits()
    {
        Properties& parameters = mTheHost.mOutputParameterList.getValueReference();

        Properties& conf = mTheHost.mConfidenceLimits.getValueReference();
        conf.clear();

        TelluriumData mat = mTheHost.mCovarianceMatrix.getValue();
        double chiReduced = mTheHost.mReducedChiSquare.getValue();
        for (int i = 0; i < parameters.count(); ++i)
        {
            double delta = 1.96 * sqrt(mat(i, i) * chiReduced);
            conf.add(new Property<double>(delta, parameters[i]->getName() + string("_confidence"), ""), true);
        }
    }

}
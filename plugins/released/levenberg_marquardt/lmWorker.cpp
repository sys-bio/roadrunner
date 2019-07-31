#pragma hdrstop
#include "lmWorker.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"
#include "rr-libstruct/lsLA.h"
#include "rrException.h"
#include "../common/telTelluriumData.h"
#include "lm.h"
#include "lib/lmmin.h"
#include "rrUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
#include "telPluginManager.h"
#include "lmObjectiveFunction.h"
//---------------------------------------------------------------------------
namespace lmfit
{
using namespace std;
using namespace rr;
using namespace tlpc;

lmWorker::lmWorker(LM& host)
:
mTheHost(host),
mRRI(NULL)
{
    memset(&mLMData, 0, sizeof(lmDataStructure));
}

bool lmWorker::isRunning() const
{
    return mThread.isRunning();
}

void lmWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            Log(lError)<<"Tried to start an already working thread!";
            return;
        }

        mThread.start(*this);    //Will spawn the run function below in a thread
    }
    else
    {
        run();
    }
}

void lmWorker::run()
{
    workerStarted();
    setupRoadRunner();

    StringList& species = mTheHost.mExperimentalDataSelectionList.getValueReference();//mMinData.getExperimentalDataSelectionList();
    Log(lInfo)<<"The following species are selected: "<<species.AsString();

    Properties& Paras =  mTheHost.mInputParameterList.getValueReference(); //mMinData.getProperties();
    Log(lInfo)<<"The following parameters are to be minimized";
    for(int i = 0; i < Paras.count(); i++)
    {
        Log(lInfo)<<Paras[i]->getName()<<" with initial value: "<<Paras[i]->getValueAsString();
    }

    mTheHost.mNrOfIter.setValue(0);
    mTheHost.mNorm.setValue(0.0);
    //Some parameters to the Algorithm..
    lm_control_struct control = lm_control_double;

    //Set defaults from Plugin
    control.ftol                    =       *(double*)  mTheHost.ftol.getValueHandle();
    control.xtol                    =       *(double*)  mTheHost.xtol.getValueHandle();
    control.gtol                    =       *(double*)  mTheHost.gtol.getValueHandle();
    control.epsilon                 =       *(double*)  mTheHost.epsilon.getValueHandle();
    control.stepbound               =       *(double*)  mTheHost.stepbound.getValueHandle();
    control.patience                =       *(int*)     mTheHost.patience.getValueHandle();

    control.msgfile = NULL;
    control.verbosity = 0;

    //Setup data structures
    setup();

    //This is the library function doing the minimization..
    lmmin(  mLMData.nrOfParameters,
            mLMData.parameters,
            mLMData.nrOfResiduePoints,
            (const void*) &mTheHost,
            lmObjectiveFunction,
            &control,
            &mTheHost.mLMStatus);

    //The user may have aborted the minization... check here..
    if(mTheHost.mTerminate)
    {
        //user did set the terminate flag to true.. discard any minimization data and get out of the
        //plugin execute code..
        Log(lInfo)<<"The minimization was terminated.. aborting";
        workerFinished();
        return;
    }

    //Post fitting data calculations
    Log(lInfo)<<"==================== Fitting Result ================================";
    Log(lInfo)<<"Nr of function evaluations: "  <<  mTheHost.mLMStatus.nfev;
    Log(lInfo)<<"Status message: "              <<  lm_infmsg[mTheHost.mLMStatus.outcome];
    Log(lInfo)<<"Minimized parameter values: ";

    mTheHost.mStatusMessage.setValueFromString(lm_infmsg[mTheHost.mLMStatus.outcome]);

    for (int i = 0; i < mLMData.nrOfParameters; ++i)
    {
        Log(lInfo)<<"Parameter "<<mLMData.parameterLabels[i]<<" = "<< mLMData.parameters[i];
    }

    Log(lInfo)<<"Norm:  "<<mTheHost.mLMStatus.fnorm;
    postFittingWork();
    workerFinished();
}

void lmWorker::postFittingWork()
{
    //Populate with data to report back
    Properties& parsOut = mTheHost.mOutputParameterList.getValueReference();
    parsOut.clear();
    for (int i = 0; i < mLMData.nrOfParameters; ++i)
    {
        parsOut.add(new Property<double>(mLMData.parameters[i], mLMData.parameterLabels[i], ""), true);
    }

    Log(lError) <<"Parameters out.."<<parsOut;
    //Set the norm property
    mTheHost.mNorm.setValue(mTheHost.mLMStatus.fnorm);

    //Create model and residuals data
    createModelData(mTheHost.mModelData.getValuePointer());
    createResidualsData(mTheHost.mResidualsData.getValuePointer());

    //Truncate Norms property to actual number of iterations
    TelluriumData tempData(mTheHost.mLMStatus.nfev, 1);
    for(int r = 0; r < tempData.rSize(); r++)
    {
        tempData(r,0) = mTheHost.rNormsData(r, 0);
    }
    mTheHost.rNormsData = tempData;
    mTheHost.rNormsData.setColumnNames(StringList("Norm"));

    //Calculate standardized residuals
    TelluriumData& residuals = *(TelluriumData*) mTheHost.mResidualsData.getValueHandle();


    //When there is a bad fit, residuals get really large and so
    //several of the statistics below overflow

    //Populate the standardized residuals
    try
    {
        TelluriumData& stdRes = *(TelluriumData*) mTheHost.mStandardizedResiduals.getValueHandle();
        stdRes = getStandardizedPopulations(residuals);

        //Create a probability plot for the residuals
        TelluriumData& probPlot = *(TelluriumData*) mTheHost.mNormalProbabilityOfResiduals.getValueHandle();
        probPlot = getNormalProbabilityPlot(stdRes);

        calculateChiSquare();
        calculateHessian();
        calculateCovariance();
        calculateConfidenceLimits();
    }
    catch(...)
    {
        Log(lError) << "There was problems calculating fit statistics";
    }

}

void lmWorker::calculateChiSquare()
{
    //Calculate ChiSquare(s)
    TelluriumData& modelData    = *(TelluriumData*) mTheHost.mModelData.getValuePointer();
    TelluriumData& obsData      = *(TelluriumData*) mTheHost.mExperimentalData.getValuePointer();
    
    Plugin* chi = mTheHost.mPM->getPlugin("tel_chisquare");

    if(!chi)
    {
        throw(Exception("Failed to get chisquare plugin in lmfit plugin"));
    }

    Property<TelluriumData>* para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
    para->setValue(obsData);

    para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
    para->setValue(modelData);

    Property<int>* intPara =  dynamic_cast< Property<int>* >(chi->getProperty("NrOfModelParameters"));
    intPara->setValue(mLMData.nrOfParameters);

    //Calculate Chi square
    chi->execute();

    Property<double>* chiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ChiSquare"));
    Property<double>* rChiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ReducedChiSquare"));

    mTheHost.mChiSquare.setValue(chiSquare->getValue());
    mTheHost.mReducedChiSquare.setValue(rChiSquare->getValue());

    Log(lInfo)<<"Chi Square = "<<chiSquare->getValue();
    Log(lInfo)<<"Reduced Chi Square = "<<rChiSquare->getValue();
}

double lmWorker::getChi(const Properties& parameters)
{
    Log(lDebug)<<"Getting chisquare using parameters: "<<parameters;
    //Reset RoadRunner
    mRRI->reset();

    for(int i = 0; i < parameters.count(); i++)
    {
        Property<double> *para = (Property<double>*) (parameters[i]);
        mRRI->setValue(para->getName(), para->getValue());
    }

    rr::SimulateOptions options;
    options.start = mLMData.timeStart;
    options.duration = mLMData.timeEnd - mLMData.timeStart;
    options.steps = mLMData.nrOfTimePoints - 1;
    mRRI->reset();

    const DoubleMatrix* modelData = NULL;
    if(mRRI->simulate(&options))
    {
        modelData = mRRI->getSimulationData();
    }

    TelluriumData& obsData      = *(TelluriumData*) mTheHost.mExperimentalData.getValuePointer();
    Plugin* chi                 = mTheHost.mPM->getPlugin("tel_chisquare");

    Property<TelluriumData>* para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
    para->setValue(obsData);

    para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
    TelluriumData td;
    td.setData(*modelData);
    para->setValue(td);

    Property<int>* intPara =  dynamic_cast< Property<int>* >(chi->getProperty("NrOfModelParameters"));
    intPara->setValue(mLMData.nrOfParameters);

    //Calculate Chi square
    chi->execute();
    Property<double>* chiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ChiSquare"));
    return chiSquare->getValue();
}

void lmWorker::calculateHessian()
{
    double eta = 6.06e-6;
    int nrOfParameters = mTheHost.mOutputParameterList.getValueReference().count();

    Properties& ref = mTheHost.mOutputParameterList.getValueReference();

    Properties copy; //Gotta add a copy/assignemnt opertor to Properties..
    for(int i = 0; i < ref.count(); i++)
    {
        copy.add(ref[i]);
    }

    //We need to calculate ChiSquares for various perturbatin around a parameter value, in order to get the Hessian
    DoubleMatrix mat(nrOfParameters, nrOfParameters);

    for(int i = 0; i < nrOfParameters; i++)
    {
        for(int j = 0; j < nrOfParameters; j++)
        {
            if(i == j)//Diagonal elements
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
                mat(i,j) = (t1 + t2 - 2.0 * t3) / (hi*hi);
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
                mat(i,j) = (t1 - t2 -t3 + t4 )/(hi*hj);

            }
            //Reset parameter values

        }
    }
    mTheHost.mHessian.setValue(mat);
}

void lmWorker::calculateCovariance()
{
    //Check if Hessain is invertible
    DoubleMatrix mat = mTheHost.mHessian.getValue();

    ls::ComplexMatrix temp(mat); //Get a complex matrix from a double one. Imag part is zero
    ls::ComplexMatrix Inv = GetInverse(temp);

    DoubleMatrix temp2(mat.RSize(), mat.CSize());
    temp2 = Inv;

    mTheHost.mCovarianceMatrix.setValue(temp2);
}

void lmWorker::calculateConfidenceLimits()
{
    Properties& parameters = mTheHost.mOutputParameterList.getValueReference();

    Properties& conf = mTheHost.mConfidenceLimits.getValueReference();
    conf.clear();

    DoubleMatrix mat = mTheHost.mCovarianceMatrix.getValue();
    double chiReduced = mTheHost.mReducedChiSquare.getValue();
    for (int i = 0; i < mLMData.nrOfParameters; ++i)
    {
        double delta = 1.96*sqrt(mat(i,i) * chiReduced);
        conf.add(new Property<double>(delta, mLMData.parameterLabels[i] + string("_confidence"), ""), true);
    }
}

void lmWorker::workerStarted()
{
    mTheHost.mIsWorking = true;
    if(mTheHost.mWorkStartedEvent)
    {
        mTheHost.mWorkStartedEvent(mTheHost.mWorkStartedData1, mTheHost.mWorkStartedData2);
    }
}

void lmWorker::workerFinished()
{
    mTheHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
    if(mTheHost.mWorkFinishedEvent)
    {
        mTheHost.mWorkFinishedEvent(mTheHost.mWorkFinishedData1, mTheHost.mWorkFinishedData2);
    }
}

bool lmWorker::setup()
{
    StringList& species         = mTheHost.mExperimentalDataSelectionList.getValueReference();
    mLMData.nrOfSpecies         = species.Count();
    Properties parameters       = mTheHost.mInputParameterList.getValue();
    mLMData.nrOfParameters      = parameters.count();
    mLMData.parameters          = new double[mLMData.nrOfParameters];
    mLMData.mLMPlugin           = static_cast<TELHandle> (&mTheHost);

    //Setup initial parameter values
    for(int i = 0; i < mLMData.nrOfParameters; i++)
    {
        Property<double> *par = (Property<double>*) parameters[i];
        if(par)
        {
            mLMData.parameters[i] = par->getValue();
        }
        else
        {
            throw("Bad stuff..");
        }
    }

    //Patience is max number of iterations
    mTheHost.rNormsData.reSize(mTheHost.patience.getValue() * (mLMData.nrOfParameters + 1), 1);

    TelluriumData& obsData             = (mTheHost.mExperimentalData.getValueReference());
    mLMData.nrOfTimePoints              = obsData.rSize();
    mLMData.timeStart                   = obsData.getTimeStart();
    mLMData.timeEnd                     = obsData.getTimeEnd();
    mLMData.nrOfResiduePoints           = mLMData.nrOfSpecies * mLMData.nrOfTimePoints;
    mLMData.time                        = new double[mLMData.nrOfTimePoints];

    mLMData.experimentalData            = new double*[mLMData.nrOfSpecies];

    if(obsData.hasWeights())
    {
        mLMData.experimentalDataWeights = new double*[mLMData.nrOfSpecies];
    }

    mLMData.speciesLabels               = new char*[mLMData.nrOfSpecies];

    //Each species data points and label
    for (int i = 0; i < mLMData.nrOfSpecies; i++)
    {
        mLMData.experimentalData[i]     = new double[mLMData.nrOfTimePoints];
        mLMData.speciesLabels[i]        = createText(species[i]);

        if(obsData.hasWeights())
        {
            mLMData.experimentalDataWeights[i] = new double[mLMData.nrOfTimePoints];
        }
    }

    //Populate Experimental Data
    for (int i = 0; i < mLMData.nrOfSpecies; i++)
    {
        for(int timePoint = 0; timePoint < mLMData.nrOfTimePoints; timePoint++)
        {
            if(i == 0)
            {
                mLMData.time[timePoint] = obsData(timePoint, 0);
            }
            mLMData.experimentalData[i][timePoint] = obsData(timePoint, i + 1);
        }
    }

    //Populate weights..
    if(obsData.hasWeights())
    {
        for (int i = 0; i < mLMData.nrOfSpecies; i++)
        {
            for(int timePoint = 0; timePoint < mLMData.nrOfTimePoints; timePoint++)
            {
                mLMData.experimentalDataWeights[i][timePoint] = obsData.getWeight(timePoint, i + 1);
            }
        }
    }

    mLMData.parameterLabels           = new char*[mLMData.nrOfParameters];
    for (int i = 0; i < mLMData.nrOfParameters; i++)
    {
        mLMData.parameterLabels[i]     = createText(parameters[i]->getName());
    }

    mLMData.roadrunner = mRRI;
    mRRI->setSelections(species);

    mLMData.mProgressEvent               = mTheHost.mWorkProgressEvent;
    mLMData.mProgressEventContextData    = mTheHost.mWorkProgressData1;

    return true;
}

bool lmWorker::setupRoadRunner()
{
    if(mRRI)
    {
        delete mRRI;
    }

    mRRI = new RoadRunner;
    mRRI->load(mTheHost.mSBML.getValue());
    mRRI->setSelections(mTheHost.getExperimentalDataSelectionList());
    return true;
}

void lmWorker::createModelData(TelluriumData* _data)
{
    TelluriumData& data = *(_data);
    //We now have the parameters
    StringList selList("time");
    selList.Append(mTheHost.mModelDataSelectionList.getValue());

    data.reSize(mLMData.nrOfTimePoints, selList.Count());
    mRRI->reset();
    mRRI->setSelections(selList);

    for(int i = 0; i < mLMData.nrOfParameters; i++)
    {
        mRRI->setValue(mLMData.parameterLabels[i], mLMData.parameters[i]);
    }

    rr::SimulateOptions options;
    options.start = mLMData.timeStart;
    options.duration = mLMData.timeEnd - mLMData.timeStart;
    options.steps = mLMData.nrOfTimePoints - 1;
    mRRI->reset();

    if(mRRI->simulate(&options))
    {
        data.setData(*mRRI->getSimulationData());
    }
}

void lmWorker::createResidualsData(TelluriumData* _data)
{
    TelluriumData& resData = *(_data);
    //We now have the parameters
    TelluriumData& obsData = (mTheHost.mExperimentalData.getValueReference());
    TelluriumData& modData = (mTheHost.mModelData.getValueReference());

    resData.reSize(modData.rSize(), modData.cSize());

    //setup coulumn names
    resData.setColumnNames(modData.getColumnNames());

    for(int sel = 0; sel < mLMData.nrOfSpecies + 1; sel++)    //selection 1 becuase of time column..
    {
        for(int i = 0; i < mLMData.nrOfTimePoints; i++)
        {
            if(sel == 0)
            {
                resData(i, sel) = modData(i, sel);    //Time
            }
            else
            {
                //The modData may contain data for other species than that was fitted..
                //We need to find out what coulmn correspond to what..
                string specie = obsData.getColumnName(sel);
                int colNr = modData.getColumnIndex(specie);
                if(colNr != -1)
                {
                    resData(i,sel) = obsData(i, sel) - modData(i, colNr);
                }
                else
                {
                    Log(lError)<<"Problem with column names when creating residual data!";
                }
            }
        }
    }
}



}

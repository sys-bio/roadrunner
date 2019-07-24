#pragma hdrstop
#include "Poco/Path.h"
#include "Poco/File.h"
#include "telTelluriumData.h"
#include "rrLogger.h"
#include "telAutoWorker.h"
#include "telAutoPlugin.h"
#include "rrUtils.h"
#include "telAutoUtils.h"
//---------------------------------------------------------------------------

using namespace rr;
using namespace telauto;

AutoWorker::AutoWorker(AutoPlugin& host)
:
mTheHost(host),
mRRAuto(mTheHost.getRRAuto())
{}

AutoWorker::~AutoWorker()
{}

bool AutoWorker::isRunning() const
{
    return mThread.isRunning();
}

bool AutoWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            Log(lError)<<"Tried to start an already working thread!";
            return false;
        }

        mThread.start(*this);    //Will spawn the run function below in a thread
    }
    else
    {
        run();
    }
    return true;
}

void AutoWorker::run()
{
    if(mTheHost.hasStartedEvent())
    {
        pair<void*, void*> passTroughData = mTheHost.getWorkStartedData();
        mTheHost.mWorkStartedEvent(passTroughData.first, passTroughData.second);    //Tell anyone who wants to know
    }

    if(!setupAuto())
    {
        Log(lError)<<"Failed to setup auto..";
        if(mTheHost.hasFinishedEvent())
        {
            pair<void*, void*> passTroughData = mTheHost.getWorkFinishedData();
            mTheHost.mWorkFinishedEvent(passTroughData.first, passTroughData.second);
        }
        return;
    }


    //==================== This is where we call auto
    mRRAuto.run();

    //Capture output data files
    string tempFolder;
    if(mTheHost.mTempFolder.getValue() == ".")
    {
        tempFolder = getCWD();
    }
    else
    {
        tempFolder = mTheHost.mTempFolder.getValue();
    }

    if(mTheHost.mCaptureOutputFiles.getValue() == true)
    {
        mTheHost.mFort2.setValue(getFileContent(joinPath(tempFolder, "fort.2")));
        mTheHost.mFort3.setValue(getFileContent(joinPath(tempFolder, "fort.3")));
        mTheHost.mFort8.setValue(getFileContent(joinPath(tempFolder, "fort.8")));
        mTheHost.mFort9.setValue(getFileContent(joinPath(tempFolder, "fort.9")));
    }

    //Fort 6 and 7 is always captured
    mTheHost.mFort6.setValue(getFileContent(joinPath(tempFolder, "fort.6")));
    //The bifurcation diagram is in fort.7, most important
    string fort7 = joinPath(tempFolder, "fort.7");
    if(!fileExists(fort7))
    {
        throw(Exception("Could not read auto data output file fort.7!"));
    }

    mTheHost.mFort7.setValue(getFileContent(fort7));

    //Cleanup after auto..
    if(mTheHost.mKeepTempFiles.getValue() == false)
    {
        removeTempFiles(tempFolder);
    }

    //Parse output data
    mAutoDataParser.parse(mTheHost.mFort7);

    //Copy data from the parser to the plugins Properties
    mTheHost.mBifurcationData.setValue(mAutoDataParser.getSolutionData());
    mTheHost.mBifurcationPoints.setValue(mAutoDataParser.getBifurcationPoints());
    mTheHost.mBifurcationLabels.setValue(mAutoDataParser.getBifurcationLabels());

    //Change Telluriumdata header to match labels in the SBML model
    vector<rr::SelectionRecord>  selRecs = mTheHost.mRR->getSteadyStateSelections();
    StringList                   selList = getRecordsAsStrings(selRecs);

    TelluriumData& data =  mTheHost.mBifurcationData.getValueReference();

    //First column is the selected parameter
    data.setColumnName(0, mTheHost.mPrincipalContinuationParameter.getValue());
    for(int col = 1; col < data.cSize(); col++)
    {
        string newName = selList[col -1 ];
        data.setColumnName(col, newName);
    }

    if(mTheHost.hasFinishedEvent())
    {
        pair<void*, void*> passTroughData = mTheHost.getWorkFinishedData();
        mTheHost.mWorkFinishedEvent(passTroughData.first, passTroughData.second);
    }
}

bool AutoWorker::setupAuto()
{
    //The following flag need to be true
    if (mTheHost.mAllowConservedMoiety.getValue() == true)
    {
        mTheHost.mRR->setConservedMoietyAnalysis(true);
    }
    else
    {
        mTheHost.mRR->setConservedMoietyAnalysis(false);
    }

    //Transfer AUTO constants to AUTO interface
    mRRAuto.assignProperties(&(mTheHost.mProperties));
    mTheHost.mRR->load(mTheHost.getSBML());

    mRRAuto.selectParameter(mTheHost.mPrincipalContinuationParameter.getValue());

    if(mTheHost.mScanDirection.getValue() == "Positive")
    {
        mRRAuto.setScanDirection(sdPositive);
    }
    else
    {
        mRRAuto.setScanDirection(sdNegative);
    }

	//mRRAuto.setPreSimulation(mTheHost.mPreSimulation.getValue());

    string str = mRRAuto.getConstantsAsString();
    Log(lInfo)<<str;
    return true;
}

bool AutoWorker::removeTempFiles(const string& tempFolder)
{
    StringList tempFiles("fort.2, fort.3, fort.8, fort.7, fort.9, fort.6");
    for(int i =0; i < tempFiles.Count(); i++)
    {
        Poco::File tempFile(joinPath(tempFolder, tempFiles[i]));
        tempFile.remove();
    }
    return true;
}

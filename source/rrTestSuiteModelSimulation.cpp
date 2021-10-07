#pragma hdrstop
#include <iomanip>
#include <map>
#include <math.h>
#include <sstream>
#include "rrLogger.h"
#include "rrTestSuiteModelSimulation.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"

using std::filesystem::path;

namespace rr
{

TestSuiteModelSimulation::TestSuiteModelSimulation(const path& dataOutputFolder, const path& modelFilePath, const path& modelFileName)
:
SBMLModelSimulation(dataOutputFolder, dataOutputFolder),
mCurrentCaseNumber(-1),
mIntegratorName(""),
mNrOfFailingPoints(0),
mLargestError(0)
{
    //make sure the output folder exists..
    mResultData.setName("ResultData");
    mReferenceData.setName("ReferenceData");
    mErrorData.setName("ErrorData");
}

TestSuiteModelSimulation::~TestSuiteModelSimulation()
{}

void TestSuiteModelSimulation::SetCaseNumber(int cNr)
{
    mCurrentCaseNumber = cNr;
}

void TestSuiteModelSimulation::SetIntegrator(std::string integrator)
{
    mIntegratorName = integrator;
}

bool TestSuiteModelSimulation::CopyFilesToOutputFolder()
{
    if(mModelSettingsFileName.empty())
    {
        mModelSettingsFileName = mModelFilePath / GetSettingsFileNameForCase(mCurrentCaseNumber);
    }

    path fName = getFileName(mModelSettingsFileName.string());
    fName = mDataOutputFolder / fName;
#if defined(WIN32)
    return CopyFileA(mModelSettingsFileName.string().c_str(), fName.string().c_str(), false) == TRUE ? true : false;
#else
    return false;
#endif
}

bool TestSuiteModelSimulation::LoadSettings(const path& settingsFName)
{
    mModelSettingsFileName = (settingsFName);

    if(mModelSettingsFileName.empty())
    {
        mModelSettingsFileName = mModelFilePath / GetSettingsFileNameForCase(mCurrentCaseNumber);
    }
    return SBMLModelSimulation::LoadSettings(mModelSettingsFileName);
}

bool TestSuiteModelSimulation::LoadSettingsEx(const path& settingsFName)
{
    mModelSettingsFileName = (settingsFName);

    if(mModelSettingsFileName.empty())
    {
        mModelSettingsFileName = mModelFilePath / GetSettingsFileNameForCase(mCurrentCaseNumber);
    }
    bool result = SBMLModelSimulation::LoadSettings(mModelSettingsFileName);

    if (mEngine)
    {
		SimulateOptions opt = SimulateOptions();
		opt.loadSBMLSettings(mModelSettingsFileName.string());
		mEngine->setSimulateOptions(opt);
		result = true;
    }
    return result;
}

bool TestSuiteModelSimulation::LoadReferenceData(path refDataFileName)
{
    //The reference data is located in the folder where the model is located
    if (refDataFileName.empty())
    {
    	refDataFileName = mModelFilePath / GetReferenceDataFileNameForCase(mCurrentCaseNumber);
    }

    if(!std::filesystem::exists(refDataFileName))
    {
        rrLog(lWarning)<<"Could not open reference data file: "<<refDataFileName;
        return false;
    }

    std::vector<std::string> lines = getLinesInFile(refDataFileName);
    if(lines.empty())
    {
        rrLog(lWarning)<<"This file is empty..";
        return false;
    }

    //Create the data..
    for(int row = 0; row < lines.size(); row++)
    {
           std::vector<std::string> recs = splitString(lines[row], ",");
        if(row == 0) //This is the header
        {
            mReferenceData.setColumnNames(recs);
            //Assign how many columns the data has
            mReferenceData.allocate(lines.size() - 1, recs.size());
        }
        else    //This is data
        {
            for(int col = 0; col < mReferenceData.cSize(); col++)
            {
                double val = toDouble(recs[col]);
                mReferenceData(row - 1,col) = val; //First line is the header..
             }
        }
    }

    return true;
}

bool TestSuiteModelSimulation::Pass()
{
    return mNrOfFailingPoints > 0 ? false : true;
}

int TestSuiteModelSimulation::NrOfFailingPoints()
{
    return mNrOfFailingPoints;
}

double TestSuiteModelSimulation::LargestError()
{
    return mLargestError;
}

bool TestSuiteModelSimulation::CreateErrorData()
{
     mResultData = GetResult();
    //Check that result data and reference data has the same dimensions
    if(mResultData.cSize() != mReferenceData.cSize() || mResultData.rSize() != mReferenceData.rSize())
    {
        mNrOfFailingPoints = mResultData.rSize();
        return false;
    }

    mErrorData.allocate(mResultData.rSize(), mResultData.cSize());

    mLargestError = 0;
    for(int row = 0; row < mResultData.rSize(); row++)
    {
        for(int col = 0; col < mResultData.cSize(); col++)
        {
            const double& actual = mResultData(row, col);
            const double& expected = mReferenceData(row,col);
            double error = fabsl(actual - expected);
            mErrorData(row, col) = error;

			double absTol = mAbsolute;
			double relTol = mRelative;
            if(error > absTol + relTol*fabs(mReferenceData(row,col)))
            {
                mNrOfFailingPoints++;;
            }

            if(error > mLargestError)
            {
                mLargestError = error;
            }
        }
    }
    return true;
}

bool TestSuiteModelSimulation::SaveAllData()
{
    //Save all data to one file that can be plotted "as one"

    //First save the reference data to a file for comparison to result data
    path refDataFileName = mDataOutputFolder / GetReferenceDataFileNameForCase(mCurrentCaseNumber);
    std::ofstream fs(refDataFileName.c_str());
    fs << mReferenceData;
    fs.close();

    std::string outputAllFileName;
    std::string dummy;
    string resultComp = "-result-comparison.csv";
    if (!mIntegratorName.empty())
    {
        resultComp = "-" + mIntegratorName + resultComp;
    }

    createTestSuiteFileNameParts(mCurrentCaseNumber, resultComp, dummy, outputAllFileName, dummy, dummy);
    fs.open(mDataOutputFolder / outputAllFileName.c_str());

    //Check matrices dimension, if they are not equal, bail..?
    if(mResultData.dimension() != mReferenceData.dimension() ||
       mResultData.dimension() != mErrorData.dimension()        ||
       mErrorData.dimension()  != mReferenceData.dimension() )
    {
        rrLog(lWarning)<<"Data dimensions are not equal, not saving to one file..";
        return false;
    }
    for(int row = 0; row < mResultData.rSize(); row++)
    {
        for(int col = 0; col < mReferenceData.cSize(); col++)
        {
            if(row == 0)
            {
                if(col == 0)
                {
                    std::vector<std::string> ref_cnames =  mReferenceData.getColumnNames();
                    for(std::vector<std::string>::iterator i = ref_cnames.begin();
                            i != ref_cnames.end(); ++i) {
                        *i = *i + "_ref";
                    }
                    fs << rr::toString(ref_cnames);
                    fs << ",";
                    std::vector<std::string> res_cnames =  mResultData.getColumnNames();
                    for(std::vector<std::string>::iterator i = ref_cnames.begin();
                            i != ref_cnames.end(); ++i) {
                        *i = *i + "_rr";
                    }
                    fs << rr::toString(res_cnames);
                    fs << ",";
                    std::vector<std::string> err_names(ref_cnames.size(), "");
                    for(int i = 0; i < err_names.size(); ++i ) {
                        err_names[i] = ref_cnames[i] + "-" + res_cnames[i];
                    }
                    fs << rr::toString(err_names);
                }
            }

            //First column is the time...
            if(col == 0)
            {
                fs << std::endl << std::setw(10)<<std::left<<std::setprecision(6)<< mReferenceData(row, col); // this is time..
            }
            else
            {
                if(row <= mReferenceData.rSize())
                {
                    fs << "," << mReferenceData(row, col);
                }
                else
                {
                    fs << "," << " ";
                }
            }
        }

        //Then the simulated data
        for(int col = 0; col < mResultData.cSize(); col++)
        {
            //First column is the time...
            if(col == 0)
            {
                fs << "," << std::setw(10)<<std::left<<std::setprecision(6)<< mResultData(row , col);
            }
            else
            {
                fs << "," << mResultData(row, col);
            }
        }

        //Then the error data
        for(int col = 0; col < mErrorData.cSize(); col++)
        {
            //First column is the time...
            if(col == 0)
            {
                fs << "," << std::setw(10)<<std::left<<std::setprecision(6)<<mErrorData(row, col); //Becuase row 0 is the header
            }
            else
            {
                fs << "," << mErrorData(row, col);
            }
        }
    }

    return true;
}

path TestSuiteModelSimulation::GetSettingsFileNameForCase(int caseNr)
{
    std::stringstream name;
    name<<std::setfill('0')<<std::setw(5)<<caseNr;
    name<<std::string("-settings.txt");        //create the "00023" subfolder format
    std::string theName = name.str();
    return theName;
}

path TestSuiteModelSimulation::GetReferenceDataFileNameForCase(int caseNr)
{
    std::stringstream name;
    name<<std::setfill('0')<<std::setw(5)<<caseNr<<"-results.csv";
    return name.str();

}

} //end of namespace



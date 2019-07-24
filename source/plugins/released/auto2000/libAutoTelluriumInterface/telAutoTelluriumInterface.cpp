#pragma hdrstop
#include <stdexcept>
#include "telAutoTelluriumInterface.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "../libAuto/vsAuto.h"
#include "rrLogger.h"
#include "rrStringList.h"
#include "rrUtils.h"
#include "telAutoUtils.h"

namespace telauto
{
using namespace rr;
using namespace autolib;

//Statics
RoadRunner*     AutoTellurimInterface::mRR                      = NULL;
Properties*     AutoTellurimInterface::mProperties              = NULL;
AutoConstants   AutoTellurimInterface::mAutoConstants;
string          AutoTellurimInterface::mPCPParameterName        = gEmptyString;
StringList      AutoTellurimInterface::mModelParameters         = StringList();
StringList      AutoTellurimInterface::mModelBoundarySpecies    = StringList();

AutoTellurimInterface::AutoTellurimInterface(RoadRunner* rr)
{
    mRR = rr;
}

AutoTellurimInterface::~AutoTellurimInterface()
{}

void AutoTellurimInterface::assignRoadRunner(RoadRunner* rrInstance)
{
    mRR = rrInstance;
}

void AutoTellurimInterface::assignProperties(Properties* props)
{
    mProperties = props;
    if(mProperties)
    {
        mAutoConstants.populateFrom(mProperties);
    }
    else
    {
        mAutoConstants.reset();
    }
}

bool AutoTellurimInterface::selectParameter(const string& para)
{
    mPCPParameterName = para;
    return false;
}

bool AutoTellurimInterface::setScanDirection(ScanDirection val)
{

    if(val == sdPositive)
    {
        mAutoConstants.DS = fabs(mAutoConstants.DS);
    }
    else
    {
        mAutoConstants.DS = -1 * fabs(mAutoConstants.DS);
    }

    mAutoConstants.mScanDirection = val;
    return true;
}

bool AutoTellurimInterface::setTempFolder(const string& fldr)
{
    if(folderExists(fldr))
    {
        mTempFolder = fldr;
        return true;
    }
    else
    {
        return false;
    }
}

string AutoTellurimInterface::getTempFolder()
{
    return mTempFolder;
}

void AutoTellurimInterface::setInitialPCPValue()
{
			double value = (mAutoConstants.mScanDirection == sdPositive) ? mAutoConstants.RL0 : mAutoConstants.RL1;

			if(mModelBoundarySpecies.Contains(mPCPParameterName))
			{
				int index = mModelBoundarySpecies.indexOf(mPCPParameterName);
				mRR->setBoundarySpeciesByIndex(index, value);
			}
			else
			{
				mRR->setValue(mPCPParameterName, value);
			}


			if(mAutoConstants.PreSimulation)
			{
				rr::BasicDictionary opt;
				opt.setItem("start", mAutoConstants.PreSimulationStart);
				opt.setItem("duration", mAutoConstants.PreSimulationDuration);
				opt.setItem("steps", mAutoConstants.PreSimulationSteps);
				opt.setItem("stiff", true);
				mRR->simulate(&opt);			// TODO: Why are two simulate calls necessary? If there is only a single call, simulation does not tend towards steady-state, even with long simulation times.
				mRR->simulate(&opt);
			}

			mRR->steadyState();
		}

void AutoTellurimInterface::run()
{
        if(!mRR)
        {
            throw(Exception("Roadrunner is NULL in AutoTelluriumInterface function run()"));
        }

        if(!setupUsingCurrentModel())
        {
            throw(Exception("Failed in Setup AutoTelluriumInterface"));
        }

        //Create fort.2 file
        string temp = getConstantsAsString();
        autolib::createFort2File(temp.c_str(), joinPath(getTempFolder(),"fort.2"));

        //Run AUTO
        CallAuto(getTempFolder());
}

bool AutoTellurimInterface::setupUsingCurrentModel()
{
    mAutoConstants.NDIM = mRR->getModel()->getNumIndFloatingSpecies() + mRR->getModel()->getNumRateRules();

    //k1,k2 etc
    mModelParameters        = mRR->getGlobalParameterIds();

    //Boundary species can be used as PCP as well
    mModelBoundarySpecies   = mRR->getBoundarySpeciesIds();

    //Set initial value of Primary continuation parameter
    setInitialPCPValue();

    setCallbackStpnt(ModelInitializationCallback);
    setCallbackFunc2(ModelFunctionCallback);
    return true;
}

//Called by Auto
int autoCallConv AutoTellurimInterface::ModelInitializationCallback(long ndim, double t, double* u, double* par)
{
	rr::ExecutableModel* theModel = mRR->getModel();

	//The continuation parameter can be a 'parameter' or a boundary species
	int numBoundaries(0), numParameters(0);

	if(mModelBoundarySpecies.indexOf(mPCPParameterName) != -1)
	{
		numBoundaries = 1;
	}

	if(mModelParameters.indexOf(mPCPParameterName) != -1)
	{
		numParameters = 1;
	}

	vector<double> boundaryValues(numBoundaries);
	vector<double> globalParameters(numParameters);

	if(numBoundaries > 0)
	{
		for (int i = 0; i < numBoundaries; i++)
		{
			int selSpecieIndex      = mModelBoundarySpecies.indexOf(mPCPParameterName);
			boundaryValues[i]       = mRR->getBoundarySpeciesByIndex(selSpecieIndex);
		}
	}

	if(numParameters > 0)
	{
		for (int i = 0; i < numParameters; i++)
		{
			int selParameter    = mModelParameters.indexOf(mPCPParameterName);
			globalParameters[i] = mRR->getGlobalParameterByIndex(selParameter);
		}
	}

	int oParaSize = numBoundaries + numParameters;
	vector<double> parameterValues(oParaSize);

	for(int i = 0; i < numBoundaries; i++)
	{
		parameterValues[i] = boundaryValues[i];
	}

	for(int i = 0; i < numParameters; i++)
	{
		parameterValues[numBoundaries + i] = globalParameters[i];
	}

	for(int i = 0; i < oParaSize; i++)
	{
		par[i] = parameterValues[i];
	}

	int    nrIndFloatingSpecies = theModel->getNumIndFloatingSpecies() + theModel->getNumRateRules();
	double* floatCon            = new double[nrIndFloatingSpecies];

	theModel->getFloatingSpeciesConcentrations(nrIndFloatingSpecies, NULL, floatCon);

	int nMin = min(nrIndFloatingSpecies, ndim);

	for(int i = 0; i < nMin; i++)
	{
		u[i] = floatCon[i];
	}

	delete[] floatCon;
	return 0;
}

void autoCallConv AutoTellurimInterface::ModelFunctionCallback(const double* oVariables, const double* par, double* oResult)
{
	//The continuation parameter can be a 'parameter' OR a boundary species
	int numBoundaries(0);
	int numParameters(0);

	if(mModelBoundarySpecies.indexOf(mPCPParameterName) != -1)
	{
		numBoundaries = 1;
	}

	if(mModelParameters.indexOf(mPCPParameterName) != -1)
	{
		numParameters = 1;
	}

	rr::ExecutableModel* theModel = mRR->getModel();
	if (numBoundaries > 0)
	{
		vector<double> oBoundary(numBoundaries);
		for(int i = 0; i < numBoundaries; i++)
		{
			oBoundary[i] = par[i];
		}

		for (int i = 0; i < numBoundaries; i++)
		{
			double val = oBoundary[i];
			int selSpecieIndex      = mModelBoundarySpecies.indexOf(mPCPParameterName);
			mRR->setBoundarySpeciesByIndex(selSpecieIndex, val);
		}
	}

	if (numParameters > 0)
	{
		//Todo memory leak
		vector<double> oParameters(numParameters);
		for(int i = 0; i < numParameters; i++)
		{
			oParameters[i] = par[i];
		}

		//Currently only one variable is supported
		mRR->setValue(mPCPParameterName, oParameters[0]);
	}

	vector<rr::SelectionRecord>  selRecs = mRR->getSteadyStateSelections();
	rr::StringList              selList = getRecordsAsStrings(selRecs);

	vector<double> variableTemp(selList.size());
	int ndim = mAutoConstants.NDIM;
	int nMin = min(selList.size(), ndim);

	for (int i = 0; i < nMin; i++)
	{
		variableTemp[i] = oVariables[i];
	}

	int  numIndFloatingSpecies  = theModel->getNumIndFloatingSpecies() + theModel->getNumRateRules();
	double* tempConc            = new double[numIndFloatingSpecies];
	if(!tempConc)
	{
		throw std::runtime_error("Failed to allocate memory in AutoTellurimInterface ModelFunction CallBack.");
	}

	for(int i = 0; i < numIndFloatingSpecies; i++)
	{
		if(i < variableTemp.size())
		{
			tempConc[i] = variableTemp[i];
		}
		else
		{
			throw("Big Problem");
		}
	}

	theModel->setFloatingSpeciesConcentrations(numIndFloatingSpecies, NULL, tempConc);
	delete [] tempConc;

	//theModel->convertToAmounts();

	double  time            = theModel->getTime();
	int     stateVecSize    = theModel->getStateVector(NULL);
	double* dydts           = new double[stateVecSize];

	theModel->getStateVectorRate(time, NULL, dydts);
	nMin = min(stateVecSize, ndim);

	for(int i = 0; i < nMin; i++)
	{
		oResult[i] = dydts[i];
	}
	delete [] dydts;
}

string AutoTellurimInterface::getConstantsAsString()
{
    return mAutoConstants.getConstantsAsString();
}

}

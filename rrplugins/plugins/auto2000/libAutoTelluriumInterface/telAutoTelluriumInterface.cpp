#pragma hdrstop
#include <stdexcept>
#include "telAutoTelluriumInterface.h"
#include "../../../source/rrExecutableModel.h"
#include "../libAuto/vsAuto.h"
#include "telLogger.h"
#include "telStringList.h"
#include "telUtils.h"
#include "telAutoUtils.h"

#include "rrplugins/core/tel_api.h"
#include "../../../wrappers/C/rrc_types.h"

namespace telauto
{
using namespace tlp;
using namespace autolib;

//Statics
rrc::RRHandle AutoTellurimInterface::mRR = NULL;		
rrc::THostInterface *AutoTellurimInterface::mHostInterface = NULL;

Properties*     AutoTellurimInterface::mProperties              = NULL;
AutoConstants   AutoTellurimInterface::mAutoConstants;
string          AutoTellurimInterface::mPCPParameterName        = gEmptyString;
StringList      AutoTellurimInterface::mModelParameters         = StringList();
StringList      AutoTellurimInterface::mModelBoundarySpecies    = StringList();

AutoTellurimInterface::AutoTellurimInterface(rrc::RRHandle rr)	
{
    mRR = rr;
}

AutoTellurimInterface::~AutoTellurimInterface()
{}

void AutoTellurimInterface::assignRoadRunner(rrc::RRHandle rrInstance,rrc::THostInterface* tHostInterface)//#::assignRoadRunner(RoadRunner* rrInstance)
{
    mRR = rrInstance;
	mHostInterface = tHostInterface;
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

	if(mModelBoundarySpecies.contains(mPCPParameterName))
	{
		unsigned int index = static_cast<unsigned int>(mModelBoundarySpecies.indexOf(mPCPParameterName));
		
		mHostInterface->setBoundarySpeciesByIndex(mRR,index, value);
	}
	else
	{
		mHostInterface->setValue(mRR,mPCPParameterName.c_str(), value);
	}


	if(mAutoConstants.PreSimulation)
	{
		// Two simulations gives better results. 
		mHostInterface->simulateEx(mRR, mAutoConstants.PreSimulationStart, mAutoConstants.PreSimulationDuration, mAutoConstants.PreSimulationSteps);
		mHostInterface->simulateEx(mRR, mAutoConstants.PreSimulationStart, mAutoConstants.PreSimulationDuration, mAutoConstants.PreSimulationSteps);
	}
	
	double* val= new double;		//just to make the call compatible else no use
	mHostInterface->steadyState(mRR,val);
	delete val;
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
	autolib::createFort2File(temp.c_str(), joinPath(getTempFolder(), "fort.2"));
    //Run AUTO
    CallAuto(getTempFolder());

}

bool AutoTellurimInterface::setupUsingCurrentModel()
{
	mAutoConstants.NDIM = mHostInterface->_getNumIndFloatingSpecies(mRR) + mHostInterface->_getNumRateRules(mRR);
    //k1,k2 etc
	rrc::RRStringArrayPtr lists= mHostInterface->getGlobalParameterIds(mRR);
	StringList list1(lists->String,lists->Count);
	mModelParameters = list1;

	lists= mHostInterface->getBoundarySpeciesIds(mRR);
	StringList list2(lists->String, lists->Count);
    //Boundary species can be used as PCP as well
    mModelBoundarySpecies   = list2;

    //Set initial value of Primary continuation parameter
    setInitialPCPValue();
    setCallbackStpnt(ModelInitializationCallback);	
    setCallbackFunc2(ModelFunctionCallback);
    return true;
}

//Called by Auto
int autoCallConv AutoTellurimInterface::ModelInitializationCallback(long ndim, double t, double* u, double* par)
{
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
		double* value = new double;
		for (int i = 0; i < numBoundaries; i++)
		{
			unsigned int selSpecieIndex = static_cast<unsigned int>(mModelBoundarySpecies.indexOf(mPCPParameterName));
			mHostInterface->getBoundarySpeciesByIndex(mRR,selSpecieIndex,value);
			boundaryValues[i] = *value;
		}
		delete value;
	}

	if(numParameters > 0)
	{
		double* value=new double;
		for (int i = 0; i < numParameters; i++)
		{
			unsigned int selParameter = static_cast<unsigned int>(mModelParameters.indexOf(mPCPParameterName));
			mHostInterface->getGlobalParameterByIndex(mRR,selParameter,value);
			globalParameters[i] = *value;
		}
		delete value;
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

	int    nrIndFloatingSpecies = mHostInterface->_getNumIndFloatingSpecies(mRR)+mHostInterface->_getNumRateRules(mRR);
	double* floatCon            = new double[nrIndFloatingSpecies];

	floatCon = (mHostInterface->getFloatingSpeciesConcentrations(mRR))->Data;

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
			unsigned int selSpecieIndex = static_cast<unsigned int>(mModelBoundarySpecies.indexOf(mPCPParameterName));
			mHostInterface->setBoundarySpeciesByIndex(mRR,selSpecieIndex, val);
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
		mHostInterface->setValue(mRR, mPCPParameterName.c_str(), oParameters[0]);
	}
		
	rrc::RRStringArrayPtr temp= mHostInterface->getSteadyStateSelectionList(mRR);
	tlp::StringList selRecs(temp->String, temp->Count);

	tlp::StringList              selList = selRecs;
	vector<double> variableTemp(selList.size());
	size_t ndim = mAutoConstants.NDIM;
	size_t nMin = min(selList.size(), ndim);

	for (int i = 0; i < nMin; i++)
	{
		variableTemp[i] = oVariables[i];
	}

	int  numIndFloatingSpecies  = mHostInterface->_getNumIndFloatingSpecies(mRR) + mHostInterface->_getNumRateRules(mRR);
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

	rrc::RRVectorPtr intermediate= new rrc::RRVector;
	intermediate->Count = numIndFloatingSpecies;
	intermediate->Data = tempConc;
	mHostInterface->setFloatingSpeciesConcentrations(mRR, intermediate);


	delete [] tempConc;

	double  time            = mHostInterface->_getTime(mRR);
	int     stateVecSize    = mHostInterface->_getStateVector(mRR);
	double* dydts           = new double[stateVecSize];

	mHostInterface->_getStateVectorRate(mRR,time,dydts);
	nMin = min(stateVecSize, ndim);

	for(int i = 0; i < nMin; i++)
	{
		oResult[i] = dydts[i];
	}
	delete[] dydts;

}

string AutoTellurimInterface::getConstantsAsString()
{
    return mAutoConstants.getConstantsAsString();
}

}

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

extern rrc::THostInterface* gHostInterface;

namespace telauto
{
using namespace tlp;
using namespace autolib;

//Statics
rrc::RRHandle AutoTellurimInterface::mRR = NULL;

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

void AutoTellurimInterface::assignRoadRunner(rrc::RRHandle rrInstance)
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

	if(mModelBoundarySpecies.contains(mPCPParameterName))
	{
		unsigned int index = static_cast<unsigned int>(mModelBoundarySpecies.indexOf(mPCPParameterName));
		
		gHostInterface->setBoundarySpeciesByIndex(mRR,index, value);
	}
	else
	{
		gHostInterface->setValue(mRR,mPCPParameterName.c_str(), value);
	}


	if(mAutoConstants.PreSimulation)
	{
		// Two simulations gives better results. 
		rrc::RRCDataPtr simulationData = gHostInterface->simulateEx(mRR, mAutoConstants.PreSimulationStart, mAutoConstants.PreSimulationDuration, mAutoConstants.PreSimulationSteps);
        if (simulationData != NULL) {
            delete[] simulationData->Weights;
            if (simulationData->ColumnHeaders != NULL) {
                for (int i = 0; i < simulationData->CSize; i++)
                    delete[] simulationData->ColumnHeaders[i];
                delete[] simulationData->ColumnHeaders;
            }
            delete[] simulationData->Data;
            delete simulationData;
        }
        simulationData = gHostInterface->simulateEx(mRR, mAutoConstants.PreSimulationStart, mAutoConstants.PreSimulationDuration, mAutoConstants.PreSimulationSteps);
        if (simulationData != NULL) {
            delete[] simulationData->Weights;
            if (simulationData->ColumnHeaders != NULL) {
                for (int i = 0; i < simulationData->CSize; i++)
                    delete[] simulationData->ColumnHeaders[i];
                delete[] simulationData->ColumnHeaders;
            }
            delete[] simulationData->Data;
            delete simulationData;
        }
	}
	
	double* val= new double;		//just to make the call compatible else no use
	gHostInterface->steadyState(mRR,val);
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
	mAutoConstants.NDIM = gHostInterface->_getNumIndFloatingSpecies(mRR) + gHostInterface->_getNumRateRules(mRR);
    //k1,k2 etc
	rrc::RRStringArrayPtr lists= gHostInterface->getGlobalParameterIds(mRR);
	StringList list1(lists->String,lists->Count);
	mModelParameters = list1;
    for (int i = 0; i < lists->Count; i++) {
        delete[] lists->String[i];
    }
    delete[] lists->String;
    delete lists;

	lists= gHostInterface->getBoundarySpeciesIds(mRR);
	if (lists) {
		StringList list2(lists->String, lists->Count);
		//Boundary species can be used as PCP as well
		mModelBoundarySpecies = list2;
        for (int i = 0; i < lists->Count; i++) {
            delete[] lists->String[i];
        }
        delete[] lists->String;
        delete lists;
	}
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
			gHostInterface->getBoundarySpeciesByIndex(mRR,selSpecieIndex,value);
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
			gHostInterface->getGlobalParameterByIndex(mRR,selParameter,value);
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

	int    nrIndFloatingSpecies = gHostInterface->_getNumIndFloatingSpecies(mRR)+gHostInterface->_getNumRateRules(mRR);
	rrc::RRVectorPtr floatCon = (gHostInterface->getFloatingSpeciesConcentrations(mRR));

	int nMin = min(nrIndFloatingSpecies, ndim);
	for(int i = 0; i < nMin; i++)
	{
		u[i] = floatCon->Data[i];
	}

    if (floatCon != NULL) {
        if (floatCon->Data != NULL) {
            delete[] floatCon->Data;
            floatCon->Data = NULL;
        }

        delete floatCon;
        floatCon = NULL;
    }

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
			gHostInterface->setBoundarySpeciesByIndex(mRR,selSpecieIndex, val);
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
		gHostInterface->setValue(mRR, mPCPParameterName.c_str(), oParameters[0]);
	}
		
	rrc::RRStringArrayPtr temp= gHostInterface->getSteadyStateSelectionList(mRR);
	tlp::StringList selRecs(temp->String, temp->Count);
    if (temp->String) {
        for (int i = 0; i < temp->Count; i++) {
            delete[] temp->String[i];
        }
        delete[] temp->String;
    }
    delete temp;

	tlp::StringList              selList = selRecs;
	vector<double> variableTemp(selList.size());
	size_t ndim = mAutoConstants.NDIM;
	size_t nMin = min(selList.size(), ndim);

	for (int i = 0; i < nMin; i++)
	{
		variableTemp[i] = oVariables[i];
	}

	int  numIndFloatingSpecies  = gHostInterface->_getNumIndFloatingSpecies(mRR) + gHostInterface->_getNumRateRules(mRR);
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

    std::unique_ptr<rrc::RRVector> intermediate = std::make_unique<rrc::RRVector>();
	intermediate->Count = numIndFloatingSpecies;
	intermediate->Data = tempConc;
	gHostInterface->setFloatingSpeciesConcentrations(mRR, intermediate.get());


	delete [] tempConc;

	double  time            = gHostInterface->_getTime(mRR);
	int     stateVecSize    = gHostInterface->_getStateVector(mRR);
	double* dydts           = new double[stateVecSize];

	gHostInterface->_getStateVectorRate(mRR,time,dydts);
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

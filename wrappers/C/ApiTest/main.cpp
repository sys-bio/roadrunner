#include <iostream>
#include <fstream>
#include "rrLogger.h"
#include "rrUtils.h"
#include "unit_test/UnitTest++.h"
#include "unit_test/XmlTestReporter.h"
#include "unit_test/TestReporterStdout.h"
#include "rrc_api.h"
#include "rrGetOptions.h"
#include "src/Args.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"

#include "Suite_TestModel.h"

using namespace std;
using namespace rr;
using namespace rrc;
using namespace UnitTest;
using std::string;

string     gTempFolder              = "";
string     gRRInstallFolder         = "";
string     gTestDataFolder          = "";
bool       gDebug                   = false;
string     gTSModelsPath;
string     gCompiler                = "";

void ProcessCommandLineArguments(int argc, char* argv[], Args& args);
bool setup(Args& args);

//call with arguments, -m"modelFilePath" -r"resultFileFolder" -t"TempFolder" -s"Suites"
int main(int argc, char* argv[])
{
    cout << "RoadRunner C API Test" << endl;

	string sbmlFilepath = "C:\\vs\\src\\roadrunner\\models\\feedback.xml";
	//string sbmlFilepath = "C:\\vs\\src\\roadrunner\\models\\bistable.xml";
	//string sbmlFilepath = "C:\\vs\\src\\roadrunner\\models\\simple.xml";
	//string sbmlFilepath = "C:\\vs\\src\\roadrunner\\models\\squareWaveModel.xml";
	//string sbmlFilepath = "C:\\Users\\Wilbert\\Desktop\\BIOMD0000000009.xml";
	//string sbmlFilepath = "C:\\Users\\Wilbert\\Desktop\\BIOMD0000000203.xml";

	RRHandle _handle = createRRInstance();
	loadSBMLFromFile(_handle, sbmlFilepath.c_str());
	if (hasError()) {
   fprintf(stderr, "Unable to load SBML file \"%s\"\n", sbmlFilepath.c_str());
   return 1;
	}


	RRStringArray *strArray;
	char *settingName, *settingDesc, *settingHint;
	char *_intgList;
	int numIntgs;
	int settingType;



	// Grab info on all implemented integrators.
	numIntgs = getNumberOfIntegrators(_handle);
	printf("Number of integrators:\t %d\n", numIntgs);
	_intgList = stringArrayToString(getListOfIntegrators(_handle));

	// set integrator to CVODE
	setIntegrator(_handle, "cvode");

	// Probe default (CVODE) integrator
	printf("%s \n", getIntegratorDescription(_handle));
	printf("%s \n", getIntegratorHint(_handle));
	printf("%d \n", getNumberOfIntegratorParameters(_handle));

	strArray = getListOfIntegratorParameterNames(_handle);
	for (int i = 0; i < strArray->Count; ++i)
	{
		settingName = strArray->String[i];
		settingDesc = getIntegratorParameterDescription(_handle, settingName);
		settingHint = getIntegratorParameterHint(_handle, settingName);
		settingType = getIntegratorParameterType(_handle, settingName);

		printf("%s\n", settingName);
		printf("Type: %d\nDescription: %s\nHint: %s\n\n", settingType, settingDesc, settingHint);
	}

	// Add Gillespie Integrator to the mix and then grab updated info on all implemented integrators.
	setIntegrator(_handle, "gillespie");
	numIntgs = getNumberOfIntegrators(_handle);
	printf("Number of integrators:\t %d\n", numIntgs);
	_intgList = stringArrayToString(getListOfIntegrators(_handle));

	// Probe Gillespie integrator
	printf("%s \n", getIntegratorDescription(_handle));
	printf("%s \n", getIntegratorHint(_handle));
	printf("%d \n", getNumberOfIntegratorParameters(_handle));

	strArray = getListOfIntegratorParameterNames(_handle);
	for (int i = 0; i < strArray->Count; ++i)
	{
		settingName = strArray->String[i];
		settingDesc = getIntegratorParameterDescription(_handle, settingName);
		settingHint = getIntegratorParameterHint(_handle, settingName);
		settingType = getIntegratorParameterType(_handle, settingName);

		printf("%s\n", settingName);
		printf("Type: %d\nDescription: %s\nHint: %s\n\n", settingType, settingDesc, settingHint);
	}

	// Simulate
	RRCDataPtr result;
	result = simulateEx(_handle, 0, 10, 100);
	printf(rrCDataToString(result));
	freeRRCData(result);
	
	char a;
	cin >> a;
    return 0;
}




/**
 * @file rrc_api.cpp
 * @brief roadRunner C API 2012
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/

#pragma hdrstop
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"
#include "rrExecutableModel.h"
#include "rrCompiler.h"
#include "rrLogger.h"
#include "rrException.h"
#include "rrVersionInfo.h"
#include "rrUtils.h"
#include "rrc_types.h"
#include "rrc_api.h"           // Need to include this before the support header..
#include "rrc_utilities.h"     //Support functions, not exposed as api functions and or data
#include "rrc_cpp_support.h"   //Support functions, not exposed as api functions and or data
#include "Integrator.h"
#include "SteadyStateSolver.h"
#include "Dictionary.h"
#include "rrConfig.h"


#if defined(_MSC_VER)
    #include <direct.h>
    #define getcwd _getcwd
    #define chdir  _chdir
#elif defined(__BORLANDC__)
      #include <dir.h>
#else
    #include <unistd.h>
#endif

/*!
 \brief macro for MAX_PATH
 \ingroup macros
*/
 #if defined(_WIN32)
    #include <windef.h>
   #define RR_MAX_PATH MAX_PATH
#else
    #include <limits.h>
    #define RR_MAX_PATH PATH_MAX
#endif
//---------------------------------------------------------------------------

namespace rrc
{

using namespace std;
using namespace rr;

static vector<string>    sel_getTime(RoadRunner* rr);
static ArrayList         sel_getFluxControlCoefficientIds(RoadRunner* rr);
static ArrayList         sel_getAvailableSteadyStateSymbols(RoadRunner* rr);
static ArrayList         sel_getAvailableTimeCourseSymbols(RoadRunner* rr);
static vector<string>    sel_getBoundarySpeciesAmountIds(RoadRunner* rr);
static vector<string>    sel_getBoundarySpeciesConcIds(RoadRunner* rr);
static ArrayList         sel_getConcentrationControlCoefficientIds(RoadRunner* rr);
static ArrayList         sel_getUnscaledConcentrationControlCoefficientIds(RoadRunner* rr);
static ArrayList         sel_getElasticityCoefficientIds(RoadRunner* rr);
static ArrayList         sel_getUnscaledElasticityCoefficientIds(RoadRunner* rr);
static vector<string>    sel_getFloatingSpeciesConcSymbols(RoadRunner* rr);
static vector<string>    sel_getBoundarySpeciesConcSymbols(RoadRunner* rr);

static vector<double> rr_getRatesOfChange(RoadRunner* rr);


// TODO: move all the depracated method here:
/*
static int rr_getNumberOfReactions(RoadRunner* r);
static double rr_getReactionRate(RoadRunner* r, const int& index);
static double getRateOfChange(const int& index);
static std::vector<std::string> rr_getRateOfChangeIds(RoadRunner* r);
static std::vector<std::string> rr_getConservedMoietyIds(RoadRunner* r);
static std::vector<double> rr_getConservedMoietyValues(RoadRunner* r);
static int rr_getNumberOfCompartments(RoadRunner* r);
static void rr_setCompartmentByIndex(RoadRunner* r, const int& index, const double& value);
static double rr_getCompartmentByIndex(RoadRunner* r, const int& index);
static std::vector<std::string> rr_getCompartmentIds(RoadRunner* r, RoadRunner* r);
static int rr_getNumberOfBoundarySpecies(RoadRunner* r);
static void rr_setBoundarySpeciesByIndex(RoadRunner* r, const int& index, const double& value);
static double rr_getBoundarySpeciesByIndex(RoadRunner* r, const int& index);
static std::vector<double> rr_getBoundarySpeciesConcentrations(RoadRunner* r);
static void rr_setBoundarySpeciesConcentrations(RoadRunner* r, const std::vector<double>& values);
static std::vector<std::string> rr_getBoundarySpeciesIds(RoadRunner* r);
static int rr_getNumberOfFloatingSpecies(RoadRunner* r);
static double rr_getFloatingSpeciesByIndex(RoadRunner* r, int index);
static void rr_setFloatingSpeciesByIndex(RoadRunner* r, int index, double value);
static std::vector<double> rr_getFloatingSpeciesConcentrations(RoadRunner* r);
static std::vector<double> rr_getFloatingSpeciesInitialConcentrations(RoadRunner* r);
static void rr_setFloatingSpeciesConcentrations(RoadRunner* r, const std::vector<double>& values);
static void rr_setFloatingSpeciesInitialConcentrationByIndex(RoadRunner* r, const int& index,
        const double& value);
static void rr_setFloatingSpeciesInitialConcentrations(RoadRunner* r, const std::vector<double>& values);
static std::vector<std::string> rr_getFloatingSpeciesIds(RoadRunner* r);
static std::vector<std::string> rr_getFloatingSpeciesInitialConditionIds(RoadRunner* r);
static int rr_getNumberOfGlobalParameters(RoadRunner* r);
static void rr_setGlobalParameterByIndex(RoadRunner* r, const int index, const double value);
static double rr_getGlobalParameterByIndex(RoadRunner* r, const int& index);
static std::vector<double> rr_getGlobalParameterValues(RoadRunner* r);
static std::vector<std::string> rr_getGlobalParameterIds(RoadRunner* r);
static int rr_getNumberOfDependentSpecies(RoadRunner* r);
static int rr_getNumberOfIndependentSpecies(RoadRunner* r);
static std::vector<double> rr_getReactionRates(RoadRunner* r);
static std::vector<std::string> rr_getReactionIds(RoadRunner* r);
 */

RRHandle rrcCallConv createRRInstance()
{
    start_try
        string rrInstallFolder(getParentFolder(getRRCAPILocation()));
        return new RoadRunner("", getTempDir(), joinPath(rrInstallFolder, "rr_support"));
    catch_ptr_macro
}

RRHandle rrcCallConv createRRInstanceEx(const char* tempFolder, const char* compiler_cstr)
{
    start_try
        char* text1 = getRRCAPILocation();
        string text2 = getParentFolder(text1);
        string rrInstallFolder(text2);
        rr::freeText(text1);
        string compiler = compiler_cstr ? compiler_cstr : "";

        if(tempFolder != NULL && !fileExists(tempFolder))
        {
            stringstream msg;
            msg<<"The temporary folder: "<<tempFolder<<" do not exist";
            Log(Logger::LOG_ERROR)<<msg.str();
            throw(Exception(msg.str()));
        }
        else if(tempFolder)
        {
            return new RoadRunner(compiler, tempFolder, joinPath(rrInstallFolder, "rr_support"));
        }
        else
        {
            return new RoadRunner(compiler, getTempDir(), joinPath(rrInstallFolder, "rr_support"));
        }
    catch_ptr_macro
}

char* rrcCallConv getInstallFolder()
{
    if(!gInstallFolder[0])
    {
        strncpy(gInstallFolder, "/usr/local", RR_MAX_PATH);
    }
    return gInstallFolder;
}

bool rrcCallConv setInstallFolder(const char* folder)
{
    start_try
        return strncpy(gInstallFolder, folder, RR_MAX_PATH) != NULL ? true : false;
    catch_bool_macro
}

char* rrcCallConv getAPIVersion()
{
    start_try
        return rr::createText("1.0");
    catch_ptr_macro
}

char* rrcCallConv getCPPAPIVersion(RRHandle handle)
{
    start_try
        return rr::createText(rr::getVersionStr());
    catch_ptr_macro
}

int rrcCallConv getVersion()
{
    int nmajor = 10000 * RR_VERSION_MAJOR;
    int nminor = 100 *  RR_VERSION_MINOR;
    int npatch = 1 * RR_VERSION_PATCH;

    return nmajor + nminor + npatch;
}

char* rrcCallConv getVersionStr()
{
    return rr::createText(rr::getVersionStr(VERSIONSTR_BASIC).c_str());
}


char* rrcCallConv getVersionEx()
{
    return rr::createText(rr::getVersionStr(VERSIONSTR_BASIC | VERSIONSTR_COMPILER | VERSIONSTR_DATE).c_str());
}

char* rrcCallConv getRRCAPILocation()
{
#if defined(_WIN32) || defined(WIN32)
    char path[MAX_PATH];
    HINSTANCE handle = NULL;
    const char* dllName = "roadrunner_c_api";
    handle = GetModuleHandle(dllName);
    int nrChars = GetModuleFileNameA(handle, path, sizeof(path));
    if(nrChars != 0)
    {
        string aPath = getFilePath(path);
        char* text = rr::createText(aPath);
        return text;
    }
    return NULL;
#else
    return rr::createText(joinPath(getInstallFolder(),"/lib"));
#endif
}

char* rrcCallConv getCopyright()
{
    start_try

        char* text = rr::createText(rr::getCopyrightStr());

        return text;
    catch_ptr_macro
}

char* rrcCallConv getInfo(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getInfo());
    catch_ptr_macro
}

char* rrcCallConv getExtendedAPIInfo()
{
    start_try
        RRHandle handle = createRRInstance();
        RoadRunner* rri = castToRoadRunner(handle);
        char* text = rr::createText(rri->getExtendedVersionInfo());
        freeRRInstance(handle);
        return text;
    catch_ptr_macro
}

char* rrcCallConv getlibSBMLVersion(RRHandle handle)
{
    start_try
        return ::strdup(rr::getVersionStr(VERSIONSTR_LIBSBML).c_str());
    catch_ptr_macro
}

char* rrcCallConv getCurrentSBML(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getCurrentSBML());
    catch_ptr_macro
}

//Flags and Options
bool rrcCallConv setComputeAndAssignConservationLaws(RRHandle handle, const bool OnOrOff)
{
    //start_try
    try {
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setConservedMoietyAnalysis(OnOrOff);
        return true;
    } catch (std::exception& e) {
        std::cerr << "Error in conserved moeity analysis " << e.what() << std::endl;
        return false;
    }
    //catch_bool_macro
}

int rrcCallConv getComputeAndAssignConservationLaws(RRHandle handle, int* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getConservedMoietyAnalysis();
        return true;
    catch_bool_macro
}

bool rrcCallConv setTempFolder(RRHandle handle, const char* folder)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setTempDir(folder);
        return true;
    catch_bool_macro
}

char* rrcCallConv getTempFolder(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getTempDir());
    catch_ptr_macro
}

bool rrcCallConv setCompiler(RRHandle handle, const char* fName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(rri->getCompiler())
        {
            return rri->getCompiler()->setCompiler(fName);
        }
        else
        {
            return false;
        }
    catch_bool_macro
}

char* rrcCallConv getCompiler(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        Compiler *compiler = rri->getCompiler();
        return strdup(compiler ? compiler->getCompiler().c_str() : "");
    catch_ptr_macro
}

bool rrcCallConv setCompilerLocation(RRHandle handle, const char* folder)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(rri->getCompiler())
        {
            return rri->getCompiler()->setCompilerLocation(folder);
        }
        else
        {
            return false;
        }
    catch_bool_macro
}

char* rrcCallConv getCompilerLocation(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getCompiler()->getCompilerLocation());
    catch_ptr_macro
}

bool rrcCallConv setSupportCodeFolder(RRHandle handle,const char* folder)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(rri->getCompiler())
        {
            return rri->getCompiler()->setSupportCodeFolder(folder);
        }
        else
        {
            return false;
        }
    catch_bool_macro
}

char* rrcCallConv getSupportCodeFolder(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getCompiler()->getSupportCodeFolder());
    catch_ptr_macro
}

char* rrcCallConv getWorkingDirectory()
{
    start_try
        return rr::createText(rr::getCWD());
    catch_ptr_macro
}

bool rrcCallConv loadSBMLFromFile(RRHandle _handle, const char* fileName)
{
    start_try
        //Check first if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

        RoadRunner* rri = castToRoadRunner(_handle);
        rri->load(fileName);
        return true;
    catch_bool_macro
}

bool rrcCallConv loadSBMLFromFileE(RRHandle _handle, const char* fileName, bool forceRecompile)
{
    start_try
        //Check first if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

        RoadRunner* rri = castToRoadRunner(_handle);

        LoadSBMLOptions opt;
        opt.modelGeneratorOpt = forceRecompile ?
                opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE :
                opt.modelGeneratorOpt & ~LoadSBMLOptions::RECOMPILE;

        rri->load(fileName, &opt);
        return true;
    catch_bool_macro
}

bool rrcCallConv loadSBML(RRHandle handle, const char* sbml)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->load(sbml);
        return true;
    catch_bool_macro
}

bool rrcCallConv loadSBMLEx(RRHandle handle, const char* sbml, bool forceRecompile)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);

        LoadSBMLOptions opt;
                opt.modelGeneratorOpt = forceRecompile ?
                        opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE :
                        opt.modelGeneratorOpt & ~LoadSBMLOptions::RECOMPILE;

        rri->load(sbml, &opt);
        return true;
    catch_bool_macro
}


bool rrcCallConv loadSimulationSettings(RRHandle handle, const char* fileName)
{
    start_try
        //Check first if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

        RoadRunner* rri = castToRoadRunner(handle);
		SimulateOptions opts = rri->getSimulateOptions();
		opts.loadSBMLSettings(fileName);
		Integrator* integrator = rri->getIntegrator();
		integrator->loadSBMLSettings(fileName);
        return true;
    catch_bool_macro
}

char* rrcCallConv getSBML(RRHandle handle)
{
    start_try
          RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getSBML());
    catch_ptr_macro
}

bool rrcCallConv isModelLoaded(RRHandle handle)
{
    start_try
          RoadRunner* rri = castToRoadRunner(handle);
        return rri->isModelLoaded();
    catch_bool_macro
}

bool rrcCallConv clearModel(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->clearModel();
    catch_bool_macro
}

bool rrcCallConv setTimeStart(RRHandle handle, const double timeStart)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->getSimulateOptions().start = timeStart;
        return true;
    catch_bool_macro
}

bool rrcCallConv setTimeEnd(RRHandle handle, const double timeEnd)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        SimulateOptions &opt = rri->getSimulateOptions();
        opt.duration = timeEnd - opt.start;
        return true;
    catch_bool_macro
}

bool rrcCallConv setNumPoints(RRHandle handle, const int nrPoints)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        SimulateOptions &opt = rri->getSimulateOptions();
        opt.steps = nrPoints - 1;
        return true;
    catch_bool_macro
}

bool rrcCallConv getTimeStart(RRHandle handle, double* timeStart)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *timeStart = rri->getSimulateOptions().start;
        return true;
    catch_bool_macro
}

bool rrcCallConv getTimeEnd(RRHandle handle, double* timeEnd)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *timeEnd = rri->getSimulateOptions().duration + rri->getSimulateOptions().start;
        return true;
    catch_bool_macro
}

bool rrcCallConv getNumPoints(RRHandle handle, int* numPoints)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *numPoints = rri->getSimulateOptions().steps + 1;
        return true;
    catch_bool_macro
}

bool rrcCallConv setTimeCourseSelectionList(RRHandle handle, const char* list)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList(list,", ");
        rri->setSelections(aList);
        return true;
    catch_bool_macro
}

RRStringArrayPtr rrcCallConv getTimeCourseSelectionList(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);

        vector<SelectionRecord> selections = rri->getSelections();

        vector<string> list(selections.size());

        for(int i = 0; i < selections.size(); ++i)
        {
            list[i] = selections[i].to_string();
        }

        StringList sNames = list;

        if(!sNames.Count())
        {
            return NULL;
        }

        return createList(sNames);
    catch_ptr_macro
}



RRCDataPtr rrcCallConv simulate(RRHandle handle)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		
	if (false)
	{
		rri = new  RoadRunner("C:/rr/source/roadrunner/models/sbml-test-suite/cases/semantic/model_editing/00017/00017-sbml-l2v4-mod.xml");
		rri->getSimulateOptions().loadSBMLSettings("C:/rr/source/roadrunner/models/sbml-test-suite/cases/semantic/model_editing/00017/00017-settings.txt");
		rri->simulate();
		RoadRunnerData resultData(rri);

		ofstream fs("C:/rr/source/roadrunner/models/sbml-test-suite/cases/semantic/model_editing/00017/00017-results.csv");
		fs << resultData;
		fs.close();
	}

	if (true)
	{
		rri = new RoadRunner();
		rri->addCompartment("compartment", 1);
		rri->addSpecies("s1", "compartment", 0.1);
		rri->addSpecies("s2", "compartment", 0.1);
		
		vector<string> reactants;
		reactants.push_back("s1");
		vector<string> products;
		products.push_back("s2");
		
		rri->addReaction("test", reactants, products, "s1");

		rri->addEvent("e1", true, "s2 > compartment");
		rri->addEventAssignment("e1", "s2", "s1 * s1");
		rri->removeSpecies("s1");

		rri->simulate();
		auto test = rri->getSimulationData();
		for (int r = 0; r < test->numRows(); r++) {
			for (int c = 0; c < test->numCols(); c++)
				std::cout << (*test)(r, c) << "\t";
			std::cout << std::endl;
		}
	
		std::cout << (rri->getSBML()) << endl;

	}


	//rri = new RoadRunner("C:/rr/source/roadrunner/models/sbml-test-suite/cases/semantic/00001/00001-sbml-l2v4.xml");
	//rri->removeReaction("reaction1");
	//rri->removeParameter("k1");
	//rri->addParameter("k1", 1);
	//rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
	
		//	rri->setKineticLaw("reaction1", "S1");
		//rri->addCompartment("compartment", 1);
		//rri->addSpecies("test", "compartment", 0.1, "substance1");
		

		rri->simulate();
		//auto test = rri->getSimulationData();
		//for (int r = 0; r < test->numRows(); r++) {
		//	for (int c = 0; c < test->numCols(); c++)
		//		std::cout << (*test)(r, c) << "\t";
		//	std::cout << std::endl;
		//}
        return createRRCData(*rri);
    catch_ptr_macro
}

RRCDataPtr rrcCallConv simulateEx(RRHandle handle, const double timeStart, const double timeEnd, const int numberOfPoints)
{
    start_try
        setTimeStart(handle, timeStart);
        setTimeEnd (handle, timeEnd);
        setNumPoints(handle, numberOfPoints);
        return simulate(handle);
    catch_ptr_macro
}

RRCDataPtr rrcCallConv getSimulationResult(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);

        //Extract the data and return struct..
        return  createRRCData(*rri);
    catch_ptr_macro
}


RRStringArrayPtr rrcCallConv getReactionIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList rNames = rri->getReactionIds();

        if(!rNames.Count())
        {
            return NULL;
        }
        return createList(rNames);
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getRatesOfChange(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> rates = rr_getRatesOfChange(rri);

        if(!rates.size())
        {
            return NULL;
        }

        return rrc::createVector(rates);
    catch_ptr_macro
}

RRStringArrayPtr rrcCallConv getRatesOfChangeIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList rNames = rri->getRateOfChangeIds();

        if(!rNames.Count())
        {
            return NULL;
        }

        return createList(rNames);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getUnscaledElasticityMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getUnscaledElasticityMatrix();
        RRDoubleMatrixPtr matrix = createMatrix(&tempMat);
        return matrix;
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getScaledElasticityMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getScaledElasticityMatrix();
        RRDoubleMatrixPtr matrix = createMatrix(&tempMat);
        return matrix;
    catch_ptr_macro
}

bool rrcCallConv getValue(RRHandle handle, const char* symbolId, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getValue(symbolId);
        return true;
    catch_bool_macro
}

bool rrcCallConv setValue(RRHandle handle, const char* symbolId, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setValue(symbolId, value);
        return true;
    catch_bool_macro
}

RRDoubleMatrixPtr rrcCallConv getStoichiometryMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ExecutableModel* model = rri->getModel();

        if(model)
        {
            int rows = 0;
            int cols = 0;

            RRDoubleMatrixPtr matrix = new RRDoubleMatrix;
            DoubleMatrix full_stoich = rri->getFullStoichiometryMatrix();

            matrix->RSize = full_stoich.numRows();
            matrix->CSize = full_stoich.numCols();
            matrix->Data =  new double[rows*cols];

            memcpy(matrix->Data, full_stoich.GetPointer(), rows*cols*sizeof(double));

            return matrix;
        }
        else
        {
            return 0;
        }
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getConservationMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getConservationMatrix();

        RRDoubleMatrixPtr matrix = new RRDoubleMatrix;
        matrix->RSize = tempMat.RSize();
        matrix->CSize = tempMat.CSize();
        matrix->Data =  new double[tempMat.RSize()*tempMat.CSize()];

        int index = 0;
        for(u_int row = 0; row < tempMat.RSize(); row++)
        {
            for(u_int col = 0; col < tempMat.CSize(); col++)
            {
                matrix->Data[index++] = tempMat(row,col);
            }
        }
        return matrix;
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getLinkMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getLinkMatrix();
        return createMatrix(&tempMat);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getNrMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getNrMatrix();
        return createMatrix(&tempMat);
    catch_ptr_macro
}

C_DECL_SPEC bool rrcCallConv hasError()
{
    return (gLastError != NULL) ? true : false;
}

char* rrcCallConv getLastError()
{
    if(!gLastError)
    {
        gLastError = rr::createText("No Error");
    }
    return gLastError;
}

int rrcCallConv getNumberOfReactions(RRHandle handle)
{
     start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfReactions();
    catch_int_macro
}

bool rrcCallConv getReactionRate(RRHandle handle, const int rateNr, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getReactionRate(rateNr);
        return true;
    catch_bool_macro
}

RRVectorPtr rrcCallConv getReactionRates(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getReactionRates();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

int rrcCallConv getNumberOfBoundarySpecies(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfBoundarySpecies();
    catch_int_macro
}

RRStringArrayPtr rrcCallConv getBoundarySpeciesIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList bNames = rri->getBoundarySpeciesIds();

        if(!bNames.Count())
        {
            return NULL;
        }

        return createList(bNames);
    catch_ptr_macro
}

RRStringArrayPtr rrcCallConv getBoundarySpeciesConcentrationIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    StringList bNames = rri->getBoundarySpeciesConcentrationIds();

    if (!bNames.Count())
    {
        return NULL;
    }

    return createList(bNames);
    catch_ptr_macro
}


int rrcCallConv getNumberOfFloatingSpecies(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfFloatingSpecies();
    catch_int_macro
}

RRStringArrayPtr rrcCallConv getFloatingSpeciesIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList fNames = rri->getFloatingSpeciesIds();

        if(!fNames.Count())
        {
            return NULL;
        }

        return createList(fNames);
    catch_ptr_macro
}

RRStringArrayPtr rrcCallConv getDependentFloatingSpeciesIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    StringList fNames = rri->getDependentFloatingSpeciesIds();

    if (!fNames.Count())
    {
        return NULL;
    }

    return createList(fNames);
    catch_ptr_macro
}


RRStringArrayPtr rrcCallConv getFloatingSpeciesConcentrationIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    StringList fNames = rri->getFloatingSpeciesConcentrationIds();

    if (!fNames.Count())
    {
        return NULL;
    }

    return createList(fNames);
    catch_ptr_macro
}


int rrcCallConv getNumberOfGlobalParameters(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfGlobalParameters();
    catch_int_macro
}

RRStringArrayPtr rrcCallConv getGlobalParameterIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList pNames = rri->getGlobalParameterIds();

        if(!pNames.Count())
        {
            return NULL;
        }

        return createList(pNames);
    catch_ptr_macro
}

bool rrcCallConv getFloatingSpeciesInitialConcentrationByIndex(RRHandle handle, int index, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ExecutableModel *model = rri->getModel();

        if (model && model->getNumFloatingSpecies() > index)
        {
            return model->getFloatingSpeciesInitConcentrations(1, &index, value) >= 0;
        }
        return true;
    catch_bool_macro
}

RRVectorPtr rrcCallConv getFloatingSpeciesConcentrations(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getFloatingSpeciesConcentrationsV();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getFloatingSpeciesAmounts(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec = rri->getFloatingSpeciesAmountsV();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getBoundarySpeciesConcentrations(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getBoundarySpeciesConcentrationsV();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getBoundarySpeciesAmounts(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec = rri->getBoundarySpeciesAmountsV();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getFloatingSpeciesInitialConcentrations(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getFloatingSpeciesInitialConcentrations();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

bool rrcCallConv setFloatingSpeciesByIndex (RRHandle handle, const int index, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setFloatingSpeciesByIndex(index, value);
        return true;
    catch_bool_macro
}

bool rrcCallConv setBoundarySpeciesByIndex (RRHandle handle, const int index, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setBoundarySpeciesByIndex(index, value);
        return true;
    catch_bool_macro
}

bool rrcCallConv setGlobalParameterByIndex(RRHandle handle, const int index, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setGlobalParameterByIndex(index, value);
        return true;
    catch_bool_macro
}

bool rrcCallConv setFloatingSpeciesInitialConcentrationByIndex(RRHandle handle, const int index, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setFloatingSpeciesInitialConcentrationByIndex(index, value);
        return true;
    catch_bool_macro
}

bool rrcCallConv setFloatingSpeciesInitialConcentrations(RRHandle handle, const RRVectorPtr vec)
{
    start_try
        vector<double> tempVec;
        copyVector(vec, tempVec);
        RoadRunner* rri = castToRoadRunner(handle);
        rri->changeInitialConditions(tempVec);
        return true;
    catch_bool_macro
}

bool rrcCallConv setFloatingSpeciesConcentrations(RRHandle handle, const RRVectorPtr vec)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);


        vector<double> tempVec;
        copyVector(vec, tempVec);
        rri->setFloatingSpeciesConcentrations(tempVec);

        return true;
    catch_bool_macro
}

bool rrcCallConv setBoundarySpeciesConcentrations(RRHandle handle, const RRVectorPtr vec)
{
    start_try
        vector<double> tempVec;
        copyVector(vec, tempVec);
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setBoundarySpeciesConcentrations(tempVec);
        return true;
    catch_bool_macro
}

bool rrcCallConv oneStep(RRHandle handle, const double currentTime, const double stepSize, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->oneStep(currentTime, stepSize);
        return true;
    catch_bool_macro
}

RRVectorPtr rrcCallConv getGlobalParameterValues(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getGlobalParameterValues();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRListPtr rrcCallConv getAvailableTimeCourseSymbols(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ArrayList slSymbols = sel_getAvailableTimeCourseSymbols(rri);
        return createArrayList(slSymbols);
    catch_ptr_macro
}

RRListPtr rrcCallConv getAvailableSteadyStateSymbols(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ArrayList slSymbols = sel_getAvailableSteadyStateSymbols(rri);
        return createArrayList(slSymbols);
    catch_ptr_macro
}

bool rrcCallConv getBoundarySpeciesByIndex (RRHandle handle, const int index, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getBoundarySpeciesByIndex(index);
        return true;
    catch_bool_macro
}

bool rrcCallConv getFloatingSpeciesByIndex (RRHandle handle, const int index, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getFloatingSpeciesByIndex(index);
        return true;
    catch_bool_macro
}

bool rrcCallConv getGlobalParameterByIndex (RRHandle handle, const int index, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getGlobalParameterByIndex(index);
        return true;
    catch_bool_macro
}

bool rrcCallConv getuCC (RRHandle handle, const char* variable, const char* parameter, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getuCC(variable, parameter);
        return true;
    catch_bool_macro
}


bool rrcCallConv getCC (RRHandle handle, const char* variable, const char* parameter, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getCC(variable, parameter);
        return true;
    catch_bool_macro
}

bool rrcCallConv getuEE(RRHandle handle, const char* name, const char* species, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getuEE(name, species);
        return true;
    catch_bool_macro
}

bool rrcCallConv getEE(RRHandle handle, const char* name, const char* species, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getEE(name, species);
        return true;
    catch_bool_macro
}

int rrcCallConv getNumberOfDependentSpecies(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfDependentSpecies();
    catch_int_macro
}

int rrcCallConv getNumberOfIndependentSpecies(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfIndependentSpecies();
    catch_int_macro
}

bool rrcCallConv steadyState(RRHandle handle, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        std::cerr << "rrcCallConv steadyState\n";
        Log(Logger::LOG_DEBUG) << "rrcCallConv steadyState ";
        *value = rri->steadyState();
        return true;
    catch_bool_macro
}

bool rrcCallConv evalModel(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->evalModel();
        return true;
    catch_bool_macro
}

char* rrcCallConv getParamPromotedSBML(RRHandle handle, const char* sArg)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        string param =  rri->getParamPromotedSBML(sArg);
        char* text = rr::createText(param.c_str());
        return text;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv computeSteadyStateValues(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getSteadyStateValues();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

bool rrcCallConv setSteadyStateSelectionList(RRHandle handle, const char* list)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList(list, " ,");
        rri->setSteadyStateSelections(aList);
        return true;
    catch_bool_macro
}

RRStringArrayPtr rrcCallConv getSteadyStateSelectionList(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<SelectionRecord>& ss = rri->getSteadyStateSelections();

        vector<string> sNames;

        for(int i = 0; i < ss.size(); ++i)
        {
            sNames.push_back(ss[i].to_string());
        }

        return createList(sNames);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getFullJacobian(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getFullJacobian();
        return createMatrix(&tempMat);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getReducedJacobian(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix tempMat = rri->getReducedJacobian();
        return createMatrix(&tempMat);
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getEigenvalues(RRHandle handle)
{
    start_try
    RoadRunner* rri = castToRoadRunner(handle);
    std::vector<ls::Complex> eigen = rri->getFullEigenValues();

    RRDoubleMatrixPtr matrix = new RRDoubleMatrix;

    matrix->RSize = eigen.size();
    matrix->CSize = 2;
    int dim =  matrix->RSize * matrix->CSize;
    if(dim)
    {
        matrix->Data =  new double[matrix->RSize * matrix->CSize];
    }
    else
    {
        delete matrix;
        return NULL;
    }

    int index = 0;
    for(u_int row = 0; row < matrix->RSize; row++)
    {
        matrix->Data[index++] = std::real(eigen[row]);
        matrix->Data[index++] = std::imag(eigen[row]);
    }
    return matrix;

    catch_ptr_macro
}

// *******  Not yet implemented  ********
// codeGenerationMode = 0 if mode is C code generation
// codeGenerationMode = 1 ig mode is internal math interpreter
bool rrcCallConv setCodeGenerationMode (RRHandle handle, int mode)
{
    return false;
}

bool rrcCallConv getScaledFloatingSpeciesElasticity(RRHandle handle, const char* reactionId, const char* speciesId, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getScaledFloatingSpeciesElasticity(reactionId, speciesId);
        return true;
    catch_bool_macro
}

RRStringArrayPtr rrcCallConv getFloatingSpeciesInitialConditionIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList = rri->getFloatingSpeciesInitialConditionIds();
        return createList(aList);
    catch_ptr_macro
}

RRStringArrayPtr rrcCallConv getFloatingSpeciesInitialConcentrationIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList = rri->getFloatingSpeciesInitialConcentrationIds();
        return createList(aList);
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getRatesOfChangeEx(RRHandle handle, const RRVectorPtr vec)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> values = rrc::createVector(vec);

        rri->setFloatingSpeciesConcentrations(values);
        vector<double> tempList = rr_getRatesOfChange(rri);
        return rrc::createVector (tempList);
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getReactionRatesEx(RRHandle handle, const RRVectorPtr vec)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> conc = rrc::createVector(vec);

        ExecutableModel *model = rri->getModel();

        if (!model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        model->setFloatingSpeciesConcentrations(conc.size(), 0, &conc[0]);

        vector<double> result(model->getNumReactions());

        model->getReactionRates(result.size(), 0, &result[0]);

        return rrc::createVector(result);
    catch_ptr_macro
}

RRListPtr rrcCallConv getElasticityCoefficientIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ArrayList aList = sel_getElasticityCoefficientIds(rri);
        RRListPtr bList = createArrayList(aList);
        return bList;
    catch_ptr_macro
}

// ----------------------------------------------------------------------
// Replacement methods for supporting solver configuration
// ----------------------------------------------------------------------

int rrcCallConv getNumRegisteredIntegrators()
{
	start_try;
		return IntegratorFactory::getInstance().getNumIntegrators();
    catch_int_macro
}

char* rrcCallConv getRegisteredIntegratorName(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredIntegratorName";
            n = 0;
        }
        return rr::createText(IntegratorFactory::getInstance().getIntegratorName(n));
    catch_ptr_macro
}

char* rrcCallConv getRegisteredIntegratorHint(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredIntegratorName";
            n = 0;
        }
        return rr::createText(IntegratorFactory::getInstance().getIntegratorHint(n));
    catch_ptr_macro
}

char* rrcCallConv getRegisteredIntegratorDescription(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredIntegratorName";
            n = 0;
        }
        return rr::createText(IntegratorFactory::getInstance().getIntegratorDescription(n));
    catch_ptr_macro
}

int rrcCallConv getNumInstantiatedIntegrators(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getExistingIntegratorNames().size();
    catch_int_macro
}

int rrcCallConv setCurrentIntegrator (RRHandle handle, char *nameOfIntegrator)
{
	start_try
        Log(Logger::LOG_DEBUG) << "setCurrentIntegrator called with " << nameOfIntegrator;
		RoadRunner* rri = castToRoadRunner(handle);
		rri->setIntegrator(std::string(nameOfIntegrator));
		return true;
		//RoadRunner* rri = castToRoadRunner(handle);
        //rri->getSimulateOptions().setItem ("integrator", nameOfIntegrator);
        //if (strcmp (topLevelSolver, listOfSteadyStateSolvers) == 0) {
        //   currentTopLevelSolver = listOfSteadyStateSolvers;
        //   return (1);
        //}
        //return true;
    catch_int_macro
}

char* rrcCallConv getCurrentIntegratorName (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       if (rri->getIntegrator()) {
            std::string str = rri->getIntegrator()->getName();
            return (rr::createText (str));
       } else {
           return rr::createText("");
       }
    catch_ptr_macro
}

char* rrcCallConv getCurrentIntegratorDescription (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
	   std::string str = rri->getIntegrator()->getDescription();
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentIntegratorHint (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
	   std::string str = rri->getIntegrator()->getHint();
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv getNumberOfCurrentIntegratorParameters (RRHandle handle)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
	    vector<std::string> keys = rri->getIntegrator()->getSettings();
		return keys.size();
    catch_int_macro
}


char* rrcCallConv getCurrentIntegratorNthParameterName (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getIntegrator()->getParamName(n);
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentIntegratorNthParameterDisplayName (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getIntegrator()->getParamDisplayName(n);
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentIntegratorNthParameterHint (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getIntegrator()->getParamHint(n);
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv getCurrentIntegratorNthParameterType (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       return (int) rri->getIntegrator()->getType(rri->getIntegrator()->getParamName(n));
    catch_int_macro
}


char* rrcCallConv getCurrentIntegratorNthParameterDescription (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getIntegrator()->getParamDesc(n);
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv resetCurrentIntegratorParameters (RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->getIntegrator()->resetSettings();
        return true;
    catch_int_macro
}

RRStringArrayPtr rrcCallConv getListOfCurrentIntegratorParameterNames (RRHandle handle)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		StringList settingsList = rri->getIntegrator()->getSettings();
		return createList(settingsList);
    catch_ptr_macro
}

char* rrcCallConv getCurrentIntegratorParameterDescription (RRHandle handle, char *parameterName)
{
    start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		std::string str = rri->getIntegrator()->getDescription(key.str());
		return (rr::createText (str));
    catch_ptr_macro
}

char* rrcCallConv getCurrentIntegratorParameterHint (RRHandle handle, char *parameterName)
{
    start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		std::string str = rri->getIntegrator()->getHint(key.str());
		return (rr::createText (str));
    catch_ptr_macro
}

int rrcCallConv getCurrentIntegratorParameterType (RRHandle handle, char *parameterName)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		return (int) rri->getIntegrator()->getType(key.str());
    catch_int_macro
}

// -------------------------------------------------------------------------------------
// Model editing methods
// -------------------------------------------------------------------------------------

bool rrcCallConv addSpecies(RRHandle handle, const char* sid, const char* compartment, double initialValue, const char* substanceUnit)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addSpecies(sid, compartment, initialValue, substanceUnit);
		return true;
	catch_bool_macro
}

bool rrcCallConv addSpeciesNoRegen(RRHandle handle, const char* sid, const char* compartment, double initialValue, const char* substanceUnit)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addSpecies(sid, compartment, initialValue, substanceUnit, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeSpecies(RRHandle handle, const char* sid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeSpecies(sid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeSpeciesNoRegen(RRHandle handle, const char* sid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeSpecies(sid, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addReactionFromSBML(RRHandle handle, const char* sbmlRep)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addReaction(sbmlRep);
		return true;
	catch_bool_macro
}

bool rrcCallConv addReactionFromSBMLNoRegen(RRHandle handle, const char* sbmlRep)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addReaction(sbmlRep, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addReaction(RRHandle handle, const char* rid, const char** reactants, int numReactants,
	const char** products, int numProducts, const char* kineticLaw)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		vector<string> r(reactants, reactants + numReactants);
		vector<string> p(products, products + numProducts);
		rri->addReaction(rid, r, p, kineticLaw);
		return true;
	catch_bool_macro
}

bool rrcCallConv addReactionNoRegen(RRHandle handle, const char* rid, const char** reactants, int numReactants,
	const char** products, int numProducts, const char* kineticLaw)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		vector<string> r(reactants, reactants + numReactants);
		vector<string> p(products, products + numProducts);
		rri->addReaction(rid, r, p, kineticLaw, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeReaction(RRHandle handle, const char* rid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeReaction(rid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeReactionNoRegen(RRHandle handle, const char* rid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeReaction(rid, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addParameter(RRHandle handle, const char* pid, double value)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addParameter(pid, value);
		return true;
	catch_bool_macro
}

bool rrcCallConv addParameterNoRegen(RRHandle handle, const char* pid, double value)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addParameter(pid, value, false);
		return true;
	catch_bool_macro
}


bool rrcCallConv removeParameter(RRHandle handle, const char* pid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeParameter(pid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeParameterNoRegen(RRHandle handle, const char* pid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeParameter(pid, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addCompartment(RRHandle handle, const char* cid, double initVolume)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addCompartment(cid, initVolume);
		return true;
	catch_bool_macro
}

bool rrcCallConv addCompartmentNoRegen(RRHandle handle, const char* cid, double initVolume)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addCompartment(cid, initVolume, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeCompartment(RRHandle handle, const char* cid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeCompartment(cid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeCompartmentNoRegen(RRHandle handle, const char* cid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeCompartment(cid, false);
		return true;
	catch_bool_macro
}


bool rrcCallConv setKineticLaw(RRHandle handle, const char* rid, const char* kineticLaw)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
	rri->setKineticLaw(rid, kineticLaw);
	return true;
	catch_bool_macro
}

bool rrcCallConv setKineticLawNoRegen(RRHandle handle, const char* rid, const char* kineticLaw)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
	rri->setKineticLaw(rid, kineticLaw, false);
	return true;
	catch_bool_macro
}


bool rrcCallConv addAssignmentRule(RRHandle handle, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addAssignmentRule(vid, formula);
		return true;
	catch_bool_macro
}

bool rrcCallConv addAssignmentRuleNoRegen(RRHandle handle, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addAssignmentRule(vid, formula, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addRateRule(RRHandle handle, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addRateRule(vid, formula);
		return true;
	catch_bool_macro
}

bool rrcCallConv addRateRuleNoRegen(RRHandle handle, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addRateRule(vid, formula, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeRules(RRHandle handle, const char* vid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeRules(vid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeRulesNoRegen(RRHandle handle, const char* vid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeRules(vid, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addEvent(RRHandle handle, const char* eid, bool useValuesFromTriggerTime, const char* trigger)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addEvent(eid, useValuesFromTriggerTime, trigger);
		return true;
	catch_bool_macro
}

bool rrcCallConv addEventNoRegen(RRHandle handle, const char* eid, bool useValuesFromTriggerTime, const char* trigger)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addEvent(eid, useValuesFromTriggerTime, trigger, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addPriority(RRHandle handle, const char* eid, const char* priority)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addPriority(eid, priority);
		return true;
	catch_bool_macro
}

bool rrcCallConv addEventNoRegenaddPriorityNoRegen(RRHandle handle, const char* eid, const char* priority)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addPriority(eid, priority, false);
		return true;
	catch_bool_macro
}


bool rrcCallConv addDelay(RRHandle handle, const char* eid, const char* delay)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addDelay(eid, delay);
		return true;
	catch_bool_macro
}

bool rrcCallConv addDelayNoRegen(RRHandle handle, const char* eid, const char* delay)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addDelay(eid, delay, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv addEventAssignment(RRHandle handle, const char* eid, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addEventAssignment(eid, vid, formula);
		return true;
	catch_bool_macro
}

bool rrcCallConv addEventAssignmentNoRegen(RRHandle handle, const char* eid, const char* vid, const char* formula)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->addEventAssignment(eid, vid, formula, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeEventAssignments(RRHandle handle, const char* eid, const char* vid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeEventAssignments(eid, vid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeEventAssignmentsNoRegen(RRHandle handle, const char* eid, const char* vid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeEventAssignments(eid, vid, false);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeEvent(RRHandle handle, const char* eid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeEvent(eid);
		return true;
	catch_bool_macro
}

bool rrcCallConv removeEventsNoRegen(RRHandle handle, const char* eid)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		rri->removeEvent(eid, false);
		return true;
	catch_bool_macro
}


// -------------------------------------------------------------------------------------
// Set and Get Methods
// -------------------------------------------------------------------------------------

int rrcCallConv getCurrentIntegratorParameterInt (RRHandle handle, char *parameterName)
{
    start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		return rri->getIntegrator()->getValueAsInt(key.str());
    catch_int_macro
}

int rrcCallConv setCurrentIntegratorParameterInt (RRHandle handle, char *parameterName, int value)
{
    start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		rri->getIntegrator()->setValue(key.str(), value);
        return true;
    catch_int_macro
}

unsigned int rrcCallConv getCurrentIntegratorParameterUInt (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return rri->getIntegrator()->getValueAsInt(key.str());
    catch_int_macro
}

int rrcCallConv setCurrentIntegratorParameterUInt (RRHandle handle, char *parameterName, unsigned int value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getIntegrator()->setValue(key.str(), value);
        return true;
    catch_int_macro
}

double rrcCallConv getCurrentIntegratorParameterDouble (RRHandle handle, char *parameterName)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		return rri->getIntegrator()->getValueAsDouble(key.str());
	catch_int_macro
}

int rrcCallConv setCurrentIntegratorParameterDouble (RRHandle handle, char *parameterName, double value)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		rri->getIntegrator()->setValue(key.str(), value);
		return true;
	catch_int_macro
}


char* rrcCallConv getCurrentIntegratorParameterString (RRHandle handle, char *parameterName)
{
    start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		std::string str = rri->getIntegrator()->getValueAsString(key.str());
		return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv setCurrentIntegratorParameterString (RRHandle handle, char *parameterName, char* value)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		rri->getIntegrator()->setValue(key.str(), value);
		return true;
	catch_int_macro
}


int rrcCallConv getCurrentIntegratorParameterBoolean (RRHandle handle, char *parameterName)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		return rri->getIntegrator()->getValueAsBool(key.str());
	catch_int_macro
}


int rrcCallConv setCurrentIntegratorParameterBoolean (RRHandle handle, char *parameterName, int value)
{
	start_try
		RoadRunner* rri = castToRoadRunner(handle);
		stringstream key;
		key << parameterName;
		rri->getIntegrator()->setValue(key.str(), (bool)value);
		return true;
	catch_int_macro
}


// ----------------------------------------------------------------------
// Replacement methods for supporting solver configuration
// ----------------------------------------------------------------------

int rrcCallConv getNumRegisteredSteadyStateSolvers()
{
    start_try;
        return SteadyStateSolverFactory::getInstance().getNumSteadyStateSolvers();
    catch_int_macro
}

char* rrcCallConv getRegisteredSteadyStateSolverName(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredSteadyStateSolverName";
            n = 0;
        }
        return rr::createText(SteadyStateSolverFactory::getInstance().getSteadyStateSolverName(n));
    catch_ptr_macro
}

char* rrcCallConv getRegisteredSteadyStateSolverHint(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredSteadyStateSolverName";
            n = 0;
        }
        return rr::createText(SteadyStateSolverFactory::getInstance().getSteadyStateSolverHint(n));
    catch_ptr_macro
}

char* rrcCallConv getRegisteredSteadyStateSolverDescription(int n)
{
    start_try;
        if (n < 0) {
            Log(Logger::LOG_WARNING) << "Negative index passed to getRegisteredSteadyStateSolverName";
            n = 0;
        }
        return rr::createText(SteadyStateSolverFactory::getInstance().getSteadyStateSolverDescription(n));
    catch_ptr_macro
}

int rrcCallConv setCurrentSteadyStateSolver (RRHandle handle, char *nameOfSteadyStateSolver)
{
    start_try
        Log(Logger::LOG_DEBUG) << "setCurrentSteadyStateSolver called with " << nameOfSteadyStateSolver;
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setSteadyStateSolver(std::string(nameOfSteadyStateSolver));
        return true;
        //RoadRunner* rri = castToRoadRunner(handle);
        //rri->getSimulateOptions().setItem ("integrator", nameOfSteadyStateSolver);
        //if (strcmp (topLevelSolver, listOfSteadyStateSolvers) == 0) {
        //   currentTopLevelSolver = listOfSteadyStateSolvers;
        //   return (1);
        //}
        //return true;
    catch_int_macro
}

char* rrcCallConv getCurrentSteadyStateSolverName (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       if (rri->getSteadyStateSolver()) {
            std::string str = rri->getSteadyStateSolver()->getName();
            return (rr::createText (str));
       } else {
           return rr::createText("");
       }
    catch_ptr_macro
}

char* rrcCallConv getCurrentSteadyStateSolverDescription (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getDescription();
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentSteadyStateSolverHint (RRHandle handle)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getHint();
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv getNumberOfCurrentSteadyStateSolverParameters (RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<std::string> keys = rri->getSteadyStateSolver()->getSettings();
        return keys.size();
    catch_int_macro
}


char* rrcCallConv getCurrentSteadyStateSolverNthParameterName (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getParamName(n);
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentSteadyStateSolverNthParameterHint (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getParamHint(n);
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv getCurrentSteadyStateSolverNthParameterType (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       return (int) rri->getSteadyStateSolver()->getType(rri->getSteadyStateSolver()->getParamName(n));
    catch_int_macro
}


char* rrcCallConv getCurrentSteadyStateSolverNthParameterDisplayName (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getParamDisplayName(n);
       return (rr::createText (str));
    catch_ptr_macro
}


char* rrcCallConv getCurrentSteadyStateSolverNthParameterDescription (RRHandle handle, int n)
{
    start_try
       RoadRunner* rri = castToRoadRunner(handle);
       std::string str = rri->getSteadyStateSolver()->getParamDesc(n);
       return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv resetCurrentSteadyStateSolverParameters (RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->getSteadyStateSolver()->resetSettings();
        return true;
    catch_int_macro
}


const char* rrcCallConv solverTypeToString (int code)
{
    switch (Variant::TypeId(code)) {
        case Variant::STRING:
            return "string";
        case Variant::BOOL:
            return "bool";
        case Variant::INT32:
            return "int32";
        case Variant::UINT32:
            return "uint32";
        case Variant::INT64:
            return "int64";
        case Variant::UINT64:
            return "uint64";
        case Variant::FLOAT:
            return "float";
        case Variant::DOUBLE:
            return "double";
        case Variant::CHAR:
            return "char";
        case Variant::UCHAR:
            return "uchar";
        case Variant::EMPTY:
            return "empty";
        default:
            return "<invalid>";
    }
}

RRStringArrayPtr rrcCallConv getListOfCurrentSteadyStateSolverParameterNames (RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList settingsList = rri->getSteadyStateSolver()->getSettings();
        return createList(settingsList);
    catch_ptr_macro
}

char* rrcCallConv getCurrentSteadyStateSolverParameterDescription (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        std::string str = rri->getSteadyStateSolver()->getDescription(key.str());
        return (rr::createText (str));
    catch_ptr_macro
}

char* rrcCallConv getCurrentSteadyStateSolverParameterHint (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        std::string str = rri->getSteadyStateSolver()->getHint(key.str());
        return (rr::createText (str));
    catch_ptr_macro
}

int rrcCallConv getCurrentSteadyStateSolverParameterType (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return (int) rri->getSteadyStateSolver()->getType(key.str());
    catch_int_macro
}

// -------------------------------------------------------------------------------------
// Set and Get Methods
// -------------------------------------------------------------------------------------

int rrcCallConv getCurrentSteadyStateSolverParameterInt (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return rri->getSteadyStateSolver()->getValueAsInt(key.str());
    catch_int_macro
}

int rrcCallConv setCurrentSteadyStateSolverParameterInt (RRHandle handle, char *parameterName, int value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getSteadyStateSolver()->setValue(key.str(), value);
        return true;
    catch_int_macro
}

unsigned int rrcCallConv getCurrentSteadyStateSolverParameterUInt (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return rri->getSteadyStateSolver()->getValueAsInt(key.str());
    catch_int_macro
}

int rrcCallConv setCurrentSteadyStateSolverParameterUInt (RRHandle handle, char *parameterName, unsigned int value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getSteadyStateSolver()->setValue(key.str(), value);
        return true;
    catch_int_macro
}

double rrcCallConv getCurrentSteadyStateSolverParameterDouble (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return rri->getSteadyStateSolver()->getValueAsDouble(key.str());
    catch_int_macro
}

int rrcCallConv setCurrentSteadyStateSolverParameterDouble (RRHandle handle, char *parameterName, double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getSteadyStateSolver()->setValue(key.str(), value);
        return true;
    catch_int_macro
}


char* rrcCallConv getCurrentSteadyStateSolverParameterString (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        std::string str = rri->getSteadyStateSolver()->getValueAsString(key.str());
        return (rr::createText (str));
    catch_ptr_macro
}


int rrcCallConv setCurrentSteadyStateSolverParameterString (RRHandle handle, char *parameterName, char* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getSteadyStateSolver()->setValue(key.str(), value);
        return true;
    catch_int_macro
}


int rrcCallConv getCurrentSteadyStateSolverParameterBoolean (RRHandle handle, char *parameterName)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        return rri->getSteadyStateSolver()->getValueAsBool(key.str());
    catch_int_macro
}


int rrcCallConv setCurrentSteadyStateSolverParameterBoolean (RRHandle handle, char *parameterName, int value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        stringstream key;
        key << parameterName;
        rri->getSteadyStateSolver()->setValue(key.str(), (bool)value);
        return true;
    catch_int_macro
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
RRStringArrayPtr rrcCallConv getEigenvalueIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList = rri->getEigenValueIds();
        return createList(aList);
    catch_ptr_macro
}

RRListPtr rrcCallConv getFluxControlCoefficientIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return createArrayList(sel_getFluxControlCoefficientIds(rri));
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getUnscaledConcentrationControlCoefficientMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix aMat = rri->getUnscaledConcentrationControlCoefficientMatrix();
        return createMatrix(&(aMat));
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getScaledConcentrationControlCoefficientMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix aMat = rri->getScaledConcentrationControlCoefficientMatrix();
        return createMatrix(&(aMat));
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getUnscaledFluxControlCoefficientMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix aMat = rri->getUnscaledFluxControlCoefficientMatrix();
        return createMatrix(&(aMat));
    catch_ptr_macro
}

RRDoubleMatrixPtr rrcCallConv getScaledFluxControlCoefficientMatrix(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        DoubleMatrix aMat = rri->getScaledFluxControlCoefficientMatrix();
        return createMatrix(&(aMat));
    catch_ptr_macro
}

static ArrayList RoadRunner_getUnscaledFluxControlCoefficientIds(RoadRunner *rr)
{
    ArrayList oResult;

    vector<string> oReactions = rr->getReactionIds();
    vector<string> oParameters = rr->getGlobalParameterIds();
    vector<string> oBoundary = rr->getBoundarySpeciesIds();
    vector<string> oConservation = rr->getConservedMoietyIds();

    for(int i = 0; i < oReactions.size(); i++)
    {
        string s = oReactions[i];

        ArrayList oCCReaction;
        vector<string> oInner;
        oCCReaction.Add(s);

        for(int i = 0; i < oParameters.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oParameters[i] + ")");
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oBoundary[i] + ")");
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oConservation[i] + ")");
        }

        oCCReaction.Add(oInner);
        oResult.Add(oCCReaction);
    }

    return oResult;
}

RRListPtr rrcCallConv getUnscaledFluxControlCoefficientIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ArrayList arrList = RoadRunner_getUnscaledFluxControlCoefficientIds(rri);
        return createArrayList(arrList);
    catch_ptr_macro
}

RRList* rrcCallConv getConcentrationControlCoefficientIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        ArrayList list = sel_getConcentrationControlCoefficientIds(rri);
        return createArrayList(list);
    catch_ptr_macro
}

RRListPtr rrcCallConv getUnscaledConcentrationControlCoefficientIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return createArrayList(sel_getUnscaledConcentrationControlCoefficientIds(rri));
    catch_ptr_macro
}

int rrcCallConv getNumberOfCompartments(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getNumberOfCompartments();
    catch_int_macro
}

bool rrcCallConv getCompartmentByIndex(RRHandle handle, const int index, double *value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getCompartmentByIndex(index);
        return true;
    catch_bool_macro
}

bool rrcCallConv setCompartmentByIndex (RRHandle handle, const int index, const double value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setCompartmentByIndex(index, value);
        return true;
    catch_bool_macro
}

RRStringArrayPtr rrcCallConv getCompartmentIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return createList(rri->getCompartmentIds());
    catch_ptr_macro
}

bool rrcCallConv getRateOfChange(RRHandle handle, const int index, double* value)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        *value = rri->getRateOfChange(index);
        return true;
    catch_bool_macro
}

char* rrcCallConv getBuildDate()
{
    return rr::createText(__DATE__);
}

char* rrcCallConv getBuildTime()
{
    return rr::createText(__TIME__);
}

char* rrcCallConv getBuildDateTime()
{
    return rr::createText(string(__DATE__) + string(" ") + string(__TIME__));
}

bool rrcCallConv freeRRInstance(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        delete rri;
        rri = NULL;
        return true;
    catch_bool_macro
}

//  Help("Returns the Symbols of all Flux Control Coefficients.")
ArrayList sel_getFluxControlCoefficientIds(RoadRunner* rr)
{
    ArrayList oResult;
    vector<string> oReactions       = rr->getReactionIds();
    vector<string> oParameters      = rr->getGlobalParameterIds();
    vector<string> oBoundary        = rr->getBoundarySpeciesIds();
    vector<string> oConservation    = rr->getConservedMoietyIds();

    for(int i = 0; i < oReactions.size(); i++)
    {
        string s = oReactions[i];

        ArrayList oCCReaction;
        StringList oInner;
        oCCReaction.Add(s);

        for(int i = 0; i < oParameters.size(); i++)
        {
            oInner.add("cc(" + s + "," + oParameters[i] + ")");
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.add("cc(" + s + "," + oBoundary[i] + ")");
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.add("cc(" + s + "," + oConservation[i] + ")");
        }

        oCCReaction.Add(oInner);
        oResult.Add(oCCReaction);
    }

    return oResult;
}

// Help(
//            "Returns symbols of the currently loaded model, that can be used for steady state analysis. format: array of arrays  { { \"groupname\", { \"item1\", \"item2\" ... } } }  or { { \"groupname\", { \"subgroup\", { \"item1\" ... } } } }."
//            )
ArrayList sel_getAvailableSteadyStateSymbols(RoadRunner* rr)
{
    ArrayList oResult;

    oResult.Add("Floating Species",                                 sel_getFloatingSpeciesConcSymbols(rr) );
    oResult.Add("Boundary Species",                                 sel_getBoundarySpeciesConcSymbols(rr) );
    oResult.Add("Floating Species (amount)",                        rr->getFloatingSpeciesIds());
    oResult.Add("Boundary Species (amount)",                        rr->getBoundarySpeciesIds());
    oResult.Add("Global Parameters",                                rr->getGlobalParameterIds());
    oResult.Add("Volumes",                                          rr->getCompartmentIds());
    oResult.Add("Fluxes",                                           rr->getReactionIds());
    oResult.Add("Flux Control Coefficients",                        sel_getFluxControlCoefficientIds(rr) );
    oResult.Add("Concentration Control Coefficients",               sel_getConcentrationControlCoefficientIds(rr) );
    oResult.Add("Unscaled Concentration Control Coefficients",      sel_getUnscaledConcentrationControlCoefficientIds(rr));
    oResult.Add("Elasticity Coefficients",                          sel_getElasticityCoefficientIds(rr) );
    oResult.Add("Unscaled Elasticity Coefficients",                 sel_getUnscaledElasticityCoefficientIds(rr) );
    oResult.Add("Eigenvalues",                                      rr->getEigenValueIds() );

    return oResult;
}

ArrayList sel_getAvailableTimeCourseSymbols(RoadRunner* rr)
{
    ArrayList oResult;
    oResult.Add("Time",                             sel_getTime(rr) );
    oResult.Add("Floating Species",                 sel_getFloatingSpeciesConcSymbols(rr) );
    oResult.Add("Boundary Species",                 sel_getBoundarySpeciesConcSymbols(rr) );
    oResult.Add("Floating Species (amount)",        rr->getFloatingSpeciesIds() );
    oResult.Add("Boundary Species (amount)",        rr->getBoundarySpeciesIds() );
    oResult.Add("Global Parameters",                rr->getGlobalParameterIds() );
    oResult.Add("Fluxes",                           rr->getReactionIds() );
    oResult.Add("Rates of Change",                  rr->getRateOfChangeIds() );
    oResult.Add("Volumes",                          rr->getCompartmentIds() );
    oResult.Add("Elasticity Coefficients",          sel_getElasticityCoefficientIds(rr) );
    oResult.Add("Unscaled Elasticity Coefficients", sel_getUnscaledElasticityCoefficientIds(rr) );
    oResult.Add("Eigenvalues",                      rr->getEigenValueIds() );
    return oResult;
}

// Help("Gets the list of boundary species amount names")
vector<string> sel_getBoundarySpeciesAmountIds(RoadRunner* rr)
{
    vector<string> result;// = new ArrayList();
    vector<string> list = rr->getBoundarySpeciesIds();
//    foreach (string s in getBoundarySpeciesIds()) oResult.add("[" + s + "]");
    for(int item = 0; item < list.size(); item++)// (object item in floatingSpeciesNames)
    {
        result.push_back(format("{0}", list[item]));
    }

    return result;
}

vector<string> sel_getBoundarySpeciesConcIds(RoadRunner* rr)
{
    vector<string> result;// = new ArrayList();
    vector<string> list = rr->getBoundarySpeciesIds();
//    foreach (string s in getBoundarySpeciesIds()) oResult.add("[" + s + "]");
    for(int item = 0; item < list.size(); item++)// (object item in floatingSpeciesNames)
    {
        result.push_back(format("[{0}]", list[item]));
    }

    return result;
}

// Help("Returns the Symbols of all Concentration Control Coefficients.")
ArrayList sel_getConcentrationControlCoefficientIds(RoadRunner* rr)
{
    ArrayList oResult;// = new ArrayList();
    vector<string> oFloating        = rr->getFloatingSpeciesIds();
    vector<string> oParameters      = rr->getGlobalParameterIds();
    vector<string> oBoundary        = rr->getBoundarySpeciesIds();
    vector<string> oConservation    = rr->getConservedMoietyIds();

    for(int i = 0; i < oFloating.size(); i++)
    {
        string s = oFloating[i];
        ArrayList oCCFloating;
        StringList oInner;
        oCCFloating.Add(s);

        for(int i = 0; i < oParameters.size(); i++)
        {
            oInner.add("cc(" + s + "," + oParameters[i] + ")");
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.add("cc(" + s + "," + oBoundary[i] + ")");
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.add("cc(" + s + "," + oConservation[i] + ")");
        }

        oCCFloating.Add(oInner);
        oResult.Add(oCCFloating);
    }

    return oResult;
}

// Help("Returns the Symbols of all Unscaled Concentration Control Coefficients.")
ArrayList sel_getUnscaledConcentrationControlCoefficientIds(RoadRunner* rr)
{
    ArrayList oResult;
    vector<string> oFloating        = rr->getFloatingSpeciesIds();
    vector<string> oParameters      = rr->getGlobalParameterIds();
    vector<string> oBoundary        = rr->getBoundarySpeciesIds();
    vector<string> oConservation    = rr->getConservedMoietyIds();

    for(int i = 0; i < oFloating.size(); i++)
    {
        string s = oFloating[i];
        ArrayList oCCFloating;
        vector<string> oInner;
        oCCFloating.Add(s);

        for(int i = 0; i < oParameters.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oParameters[i] + ")");
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oBoundary[i] + ")");
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.push_back("ucc(" + s + "," + oConservation[i] + ")");
        }

        oCCFloating.Add(oInner);
        oResult.Add(oCCFloating);
    }

    return oResult;
}

// Help("Returns the Symbols of all Elasticity Coefficients.")
ArrayList sel_getElasticityCoefficientIds(RoadRunner* rr)
{
    ArrayList oResult;
    vector<string> reactionNames        = rr->getReactionIds();
    vector<string> floatingSpeciesNames = rr->getFloatingSpeciesIds();
    vector<string> boundarySpeciesNames = rr->getBoundarySpeciesIds();
    vector<string> conservationNames    = rr->getConservedMoietyIds();
    vector<string> globalParameterNames = rr->getGlobalParameterIds();

    for(int i = 0; i < reactionNames.size(); i++)
    {
        string reac_name = reactionNames[i];
        ArrayList oCCReaction;
        oCCReaction.Add(reac_name);
        StringList oInner;

        for(int j = 0; j < floatingSpeciesNames.size(); j++)
        {
            oInner.add(format("ee({0},{1})", reac_name, floatingSpeciesNames[j]));
        }

        for(int j = 0; j < boundarySpeciesNames.size(); j++)
        {
            oInner.add(format("ee({0},{1})", reac_name, boundarySpeciesNames[j]));
        }

        for(int j = 0; j < globalParameterNames.size(); j++)
        {
            oInner.add(format("ee({0},{1})", reac_name, globalParameterNames[j]));
        }

        for(int j = 0; j < conservationNames.size(); j++)
        {
            oInner.add(format("ee({0},{1})", reac_name, conservationNames[j]));
        }

        oCCReaction.Add(oInner);
        oResult.Add(oCCReaction);
    }

    return oResult;
}

// Help("Returns the Symbols of all Unscaled Elasticity Coefficients.")
ArrayList sel_getUnscaledElasticityCoefficientIds(RoadRunner* rr)
{
    ArrayList oResult;
    vector<string> oReactions( rr->getReactionIds() );
    vector<string> oFloating = rr->getFloatingSpeciesIds();
    vector<string> oBoundary = rr->getBoundarySpeciesIds();
    vector<string> oGlobalParameters = rr->getGlobalParameterIds();
    vector<string> oConservation = rr->getConservedMoietyIds();

    for(int i = 0; i < oReactions.size(); i++)
    {
        string reac_name = oReactions[i];
        ArrayList oCCReaction;
        StringList oInner;
        oCCReaction.Add(reac_name);

        for(int j = 0; j < oFloating.size(); j++)
        {
            string variable = oFloating[j];
            oInner.add(format("uee({0},{1})", reac_name, variable));
        }

        for(int j = 0; j < oBoundary.size(); j++)
        {
            string variable = oBoundary[j];
            oInner.add(format("uee({0},{1})", reac_name, variable));
        }

        for(int j = 0; j < oGlobalParameters.size(); j++)
        {
            string variable = oGlobalParameters[j];
            oInner.add(format("uee({0},{1})", reac_name, variable));
        }

        for(int j = 0; j < oConservation.size(); j++)
        {
            string variable = oConservation[j];
            oInner.add(format("uee({0},{1})", reac_name, variable));
        }

        oCCReaction.Add(oInner);
        oResult.Add(oCCReaction);
    }

    return oResult;
}

vector<string> sel_getTime(RoadRunner* rr)
{
    vector<string> ids = rr->getFloatingSpeciesIds();
    vector<string> result;

    result.push_back("time");

    return result;
}

vector<string> sel_getFloatingSpeciesConcSymbols(RoadRunner* rr)
{
    vector<string> ids = rr->getFloatingSpeciesIds();
    vector<string> result;

    for(int i = 0; i < ids.size(); i++)
    {
        result.push_back("[" + ids[i] + "]");
    }

    return result;
}


vector<string> sel_getBoundarySpeciesConcSymbols(RoadRunner* rr)
{
    vector<string> ids = rr->getBoundarySpeciesIds();
    vector<string> result;

    for(int i = 0; i < ids.size(); i++)
    {
        result.push_back("[" + ids[i] + "]");
    }

    return result;
}

vector<double> rr_getRatesOfChange(RoadRunner* rr)
{
    ExecutableModel *mModel = rr->getModel();
    if (!mModel)
    {
        throw CoreException(gEmptyModelMessage);
    }

    mModel->computeAllRatesOfChange();
    vector<double> result(mModel->getNumFloatingSpecies());
    mModel->getFloatingSpeciesAmountRates(result.size(), 0, &result[0]);
    return result;
}

C_DECL_SPEC bool rrcCallConv getSeed(RRHandle h, long* result) {
    start_try
        RoadRunner *r = (RoadRunner*)h;
        //Integrator *intg = r->getIntegrator(Integrator::GILLESPIE);
		//*result = intg->getItem("seed").convert<long>();
		*result = r->getIntegrator()->getValue("seed").convert<long>();
        return true;
    catch_bool_macro;
}

C_DECL_SPEC bool rrcCallConv setSeed(RRHandle h, long result) {
    start_try
        RoadRunner *r = (RoadRunner*)h;
        //Integrator *intg = r->getIntegrator(Integrator::GILLESPIE);
        //intg->setItem("seed", result);
		Integrator *intg = r->getIntegrator();
		if (intg->getName() == "gillespie")
		{
			intg->setValue("seed", result);
		}
		else
		{
			Integrator *intg = IntegratorFactory::getInstance().New("gillespie", r->getModel());
			intg->setValue("seed", result);
		}
        return true;
    catch_bool_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespie(RRHandle handle) {
    start_try
        RoadRunner *r = (RoadRunner*)handle;
        SimulateOptions& o = r->getSimulateOptions();
        //o.integrator = Integrator::GILLESPIE;
        //o.integratorFlags |= Integrator::VARIABLE_STEP;
		r->setIntegrator("gillespie");
		r->getIntegrator()->setValue("variable_step_size", false);
        r->simulate();
        return createRRCData(*r);
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieEx(RRHandle handle, double timeStart, double timeEnd) {
    start_try
        setTimeStart (handle, timeStart);
        setTimeEnd (handle, timeEnd);
        return gillespie(handle);
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieOnGrid(RRHandle handle) {
    start_try
        RoadRunner *r = (RoadRunner*)handle;
        SimulateOptions& o = r->getSimulateOptions();
        //o.integrator = Integrator::GILLESPIE;
        //o.integratorFlags &= !Integrator::VARIABLE_STEP;
        r->setIntegrator("gillespie");
		r->getIntegrator()->setValue("variable_step_size", false);
        r->simulate();
        return createRRCData(*r);
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieOnGridEx(RRHandle handle, double timeStart, double timeEnd, int numberOfPoints) {
    start_try
        setTimeStart (handle, timeStart);
        setTimeEnd (handle, timeEnd);
        setNumPoints (handle, numberOfPoints);
        return gillespieOnGrid(handle);
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieMeanOnGrid(RRHandle handle, int numberOfSimulations) {
    start_try
        // Standard gillespieOnGrid setup
        RoadRunner *r = (RoadRunner*)handle;
        SimulateOptions& o = r->getSimulateOptions();
        //o.integrator = Integrator::GILLESPIE;
        //o.integratorFlags &= !Integrator::VARIABLE_STEP;
        r->setIntegrator("gillespie");
		r->getIntegrator()->setValue("variable_step_size", false);

        double steps = o.steps;

        RoadRunner &rref = const_cast<RoadRunner&>(*r);
        const DoubleMatrix& reference = *rref.getSimulationData();

        // Initializes a DoubleMatrix "res" with all zeroes
        DoubleMatrix avg(reference.RSize(), reference.CSize());
        for (int row = 0; row < reference.RSize(); row++) {
            for (int col = 0; col < reference.CSize(); col++) {
                avg(row, col) = 0;
            }
        }

        // Runs simulations, obtaining avg with Welford's Algorithm
        for (int i = 0; i < numberOfSimulations; i++) {
            const DoubleMatrix &temp = *r->simulate();
            for (int j = 0; j < steps + 1; j++) {
                avg(j, 0) += (temp(j, 0) - avg(j, 0)) / (i + 1);
            }
        }

        DoubleMatrix &result = const_cast<DoubleMatrix&>(avg);

        RRCData* rrCData = new RRCData;
        memset(rrCData, 0, sizeof(RRCData));

        // Create column info
        const std::vector<SelectionRecord> &selections = rref.getSelections();
        rrCData->ColumnHeaders = new char*[selections.size()];
        for (int i = 0; i < selections.size(); i++) {
            rrCData->ColumnHeaders[i] = rr::createText(selections[i].to_string());
        }

        // Basic Attributes of RRCData
        rrCData->RSize = result.RSize();
        rrCData->CSize = result.CSize();
        int size = rrCData->RSize * rrCData->CSize;
        rrCData->Data = new double[size];

        int index = 0;
        for (int row = 0; row < rrCData->RSize; row++) {
            for (int col = 0; col < rrCData->CSize; col++) {
                rrCData->Data[index++] = result(row, col);
            }
        }
        return rrCData;
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieMeanOnGridEx(RRHandle handle, double timeStart, double timeEnd, int numberOfPoints, int numberOfSimulations) {
    start_try
        setTimeStart (handle, timeStart);
        setTimeEnd (handle, timeEnd);
        setNumPoints (handle, numberOfPoints);
        return gillespieMeanOnGrid(handle, numberOfSimulations);
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieMeanSDOnGrid(RRHandle handle, int numberOfSimulations) {
    start_try
        // Standard gillespieOnGrid setup
        RoadRunner *r = (RoadRunner*)handle;
        SimulateOptions& o = r->getSimulateOptions();
        r->setIntegrator("gillespie");
		//o.integratorFlags &= !Integrator::VARIABLE_STEP;
		r->getIntegrator()->setValue("variable_step_size", false);

        double steps = o.steps;

        RoadRunner &rref = const_cast<RoadRunner&>(*r);
        const DoubleMatrix& reference = *rref.getSimulationData();

        // Initializes a DoubleMatrix "res" with all zeroes
        DoubleMatrix avg(reference.RSize(), reference.CSize());
        DoubleMatrix stddev(reference.RSize(), reference.CSize());
        for (int row = 0; row < reference.RSize(); row++) {
            for (int col = 0; col < reference.CSize(); col++) {
                avg(row, col) = 0;
                stddev(row, col) = 0;
            }
        }

        // Runs simulations, obtain avg and stddev with Welford's Algorithm
        for (int i = 0; i < numberOfSimulations; i++) {
            const DoubleMatrix &temp = *r->simulate();
            for (int j = 0; j < steps + 1; j++) {
                double delta = temp(j, 0) - avg(j, 0);
                avg(j, 0) += delta / (i + 1);
                stddev(j, 0) += delta * (temp(j, 0) - avg(j, 0));
            }
        }

        DoubleMatrix &result = const_cast<DoubleMatrix&>(avg);

        RRCData* rrCData = new RRCData;
        memset(rrCData, 0, sizeof(RRCData));

        // Create column info
        const std::vector<SelectionRecord> &selections = rref.getSelections();
        rrCData->ColumnHeaders = new char*[selections.size()];
        for (int i = 0; i < selections.size(); i++) {
            rrCData->ColumnHeaders[i] = rr::createText(selections[i].to_string());
        }

        // Basic Attributes of RRCData
        rrCData->RSize = result.RSize();
        rrCData->CSize = result.CSize();
        int size = rrCData->RSize * rrCData->CSize;
        rrCData->Data = new double[size];
        rrCData->Weights = new double[size];

        int index = 0;
        for (int row = 0; row < rrCData->RSize; row++) {
            for (int col = 0; col < rrCData->CSize; col++) {
                rrCData->Data[index] = result(row, col);
                rrCData->Weights[index] = result(row, col);
                index++;
            }
        }
        return rrCData;
    catch_ptr_macro
}

C_DECL_SPEC RRCDataPtr rrcCallConv gillespieMeanSDOnGridEx(RRHandle handle, double timeStart, double timeEnd, int numberOfPoints, int numberOfSimulations) {
    start_try
        setTimeStart (handle, timeStart);
        setTimeEnd (handle, timeEnd);
        setNumPoints (handle, numberOfPoints);
        return gillespieMeanSDOnGrid(handle, numberOfSimulations);
    catch_ptr_macro
}

bool rrcCallConv reset(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->reset();
        return true;
    catch_bool_macro
}

bool rrcCallConv resetAll(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->reset(SelectionRecord::TIME | SelectionRecord::RATE | SelectionRecord::FLOATING | SelectionRecord::GLOBAL_PARAMETER);
        return true;
    catch_bool_macro
}

bool rrcCallConv resetToOrigin(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->reset(SelectionRecord::ALL);
        return true;
    catch_bool_macro
}

bool rrcCallConv resetParameter(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    rri->reset(SelectionRecord::GLOBAL_PARAMETER);
    return true;
    catch_bool_macro
}

int rrcCallConv setConfigBool(const char* key, int value) {
    start_try
        rr::Config::setValue(rr::Config::stringToKey(key), (bool)value);
        return true;
    catch_int_macro
}

int rrcCallConv getConfigBool(const char* key) {
    start_try
        return rr::Config::getValue(rr::Config::stringToKey(key));
    catch_int_macro
}

int rrcCallConv setConfigInt(const char* key, int value) {
    start_try
        rr::Config::setValue(rr::Config::stringToKey(key), value);
        return true;
    catch_int_macro
}

int rrcCallConv getConfigInt(const char* key) {
    start_try
        return rr::Config::getValue(rr::Config::stringToKey(key));
    catch_int_macro
}

int rrcCallConv setConfigDouble(const char* key, double value) {
    start_try
        rr::Config::setValue(rr::Config::stringToKey(key), value);
        return true;
    catch_int_macro
}

double rrcCallConv getConfigDouble(const char* key) {
    start_try
        return rr::Config::getValue(rr::Config::stringToKey(key));
    catch_int_macro
}

RRStringArrayPtr rrcCallConv getListOfConfigKeys()
{
    start_try
        vector<string> list = Config::getKeyList();

        StringList sNames = list;

        if(!sNames.Count())
        {
            return NULL;
        }

        return createList(sNames);
    catch_ptr_macro
}

}

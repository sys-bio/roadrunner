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
#include "rrExecutableModel.h"
#include "rrCompiler.h"
#include "rrLogger.h"
#include "rrException.h"
#include "rrVersionInfo.h"
#include "rrUtils.h"
#include "rrc_api.h"           // Need to include this before the support header..
#include "rrc_utilities.h"   //Support functions, not exposed as api functions and or data
#include "rrc_cpp_support.h"   //Support functions, not exposed as api functions and or data


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
        return rr::createText("0.99");
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
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setConservedMoietyAnalysis(OnOrOff);
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
        rri->setSimulateOptions(SimulateOptions(fileName));
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

bool rrcCallConv unLoadModel(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->unLoadModel();
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

RRDataHandle rrcCallConv simulate(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);

        rri->getSimulateOptions().flags |= SimulateOptions::RESET_MODEL;
        rri->simulate();
        return (RRDataHandle) rri->getSimulationResult();

    catch_ptr_macro
}

RRDataHandle rrcCallConv simulateEx(RRHandle handle, const double timeStart, const double timeEnd, const int numberOfPoints)
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

RRDataHandle rrcCallConv getRoadRunnerData(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rri->getSimulationResult();
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
            double* data = 0;

            model->getStoichiometryMatrix(&rows, &cols, &data);

            RRDoubleMatrixPtr matrix = new RRDoubleMatrix;
            matrix->RSize = rows;
            matrix->CSize = cols;
            matrix->Data =  new double[rows*cols];

            memcpy(matrix->Data, data, rows*cols*sizeof(double));

            free(data);

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

bool rrcCallConv reset(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->reset();
        return true;
    catch_bool_macro
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
        vector<double> vec =  rri->getFloatingSpeciesConcentrations();
        RRVector* aVec = rrc::createVector(vec);
        return aVec;
    catch_ptr_macro
}

RRVectorPtr rrcCallConv getBoundarySpeciesConcentrations(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        vector<double> vec =  rri->getBoundarySpeciesConcentrations();
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
        DoubleMatrix tempMat = rri->getEigenvalues();
        return createMatrix(&tempMat);
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

bool rrcCallConv setConfigurationXML(RRHandle handle, const char* caps)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(!caps)
        {
            return false;
        }
        rri->setConfigurationXML(caps);
        return true;
    catch_bool_macro
}

char* rrcCallConv getConfigurationXML(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return rr::createText(rri->getConfigurationXML());
    catch_ptr_macro
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
RRStringArrayPtr rrcCallConv getEigenvalueIds(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        StringList aList = rri->getEigenvalueIds();
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
    oResult.Add("Eigenvalues",                                      rr->getEigenvalueIds() );

    return oResult;
}

ArrayList sel_getAvailableTimeCourseSymbols(RoadRunner* rr)
{
    ArrayList oResult;
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
    oResult.Add("Eigenvalues",                      rr->getEigenvalueIds() );
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
            oInner.add("CC:" + s + "," + oParameters[i]);
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.add("CC:" + s + "," + oBoundary[i]);
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.add("CC:" + s + "," + oConservation[i]);
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
            oInner.push_back("uCC:" + s + "," + oParameters[i]);
        }

        for(int i = 0; i < oBoundary.size(); i++)
        {
            oInner.push_back("uCC:" + s + "," + oBoundary[i]);
        }

        for(int i = 0; i < oConservation.size(); i++)
        {
            oInner.push_back("uCC:" + s + "," + oConservation[i]);
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
            oInner.add(format("EE:{0},{1}", reac_name, floatingSpeciesNames[j]));
        }

        for(int j = 0; j < boundarySpeciesNames.size(); j++)
        {
            oInner.add(format("EE:{0},{1}", reac_name, boundarySpeciesNames[j]));
        }

        for(int j = 0; j < globalParameterNames.size(); j++)
        {
            oInner.add(format("EE:{0},{1}", reac_name, globalParameterNames[j]));
        }

        for(int j = 0; j < conservationNames.size(); j++)
        {
            oInner.add(format("EE:{0},{1}", reac_name, conservationNames[j]));
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
            oInner.add(format("uEE:{0},{1}", reac_name, variable));
        }

        for(int j = 0; j < oBoundary.size(); j++)
        {
            string variable = oBoundary[j];
            oInner.add(format("uEE:{0},{1}", reac_name, variable));
        }

        for(int j = 0; j < oGlobalParameters.size(); j++)
        {
            string variable = oGlobalParameters[j];
            oInner.add(format("uEE:{0},{1}", reac_name, variable));
        }

        for(int j = 0; j < oConservation.size(); j++)
        {
            string variable = oConservation[j];
            oInner.add(format("uEE:{0},{1}", reac_name, variable));
        }

        oCCReaction.Add(oInner);
        oResult.Add(oCCReaction);
    }

    return oResult;
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

}

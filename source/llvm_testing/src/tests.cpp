#include "tests.h"
#include "llvm/ModelGeneratorContext.h"
#include "llvm/LLVMModelDataSymbols.h"
#include "llvm/ModelDataIRBuilder.h"
#include "rrException.h"
#include "rrUtils.h"
#include "llvm/LLVMIncludes.h"
#include "llvm/AssignmentRuleEvaluator.h"
#include "llvm/EvalInitialConditionsCodeGen.h"
#include "rrSparse.h"
#include "rrLogger.h"





#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/SBMLReader.h>

#include "llvm/LLVMModelData.h"

#include <utility>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "cpplapack.h"

using namespace std;
using namespace rr;
using namespace llvm;
using namespace libsbml;
using namespace rrllvm;



string getModelFileName(const string& version, int caseNumber)
{
    string dummy;
    string logFileName;
    string settingsFileName;

    //Create a log file name
    createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName,
            settingsFileName);

    //Read SBML models.....
    string home = getenv("HOME");
    string modelFilePath = home + "/src/sbml_test/";

    modelFilePath = joinPath(joinPath(modelFilePath, "cases"), "semantic");
    string modelFileName;

    createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml",
            modelFilePath, modelFileName, settingsFileName);

    modelFileName = joinPath(modelFilePath, modelFileName);

    return modelFileName;
}

bool runInitialValueAssigmentTest(const string& version, int caseNumber)
{
    LLVMModelData *md = 0;
    string modelFileName = getModelFileName(version, caseNumber);

    SBMLDocument *doc = readSBMLFromFile(modelFileName.c_str());

    try
    {

        ModelGeneratorContext c(doc, true);

        c.getModelDataSymbols().print();

        createModelData(c.getModelDataSymbols(), 0);

        ExecutionEngine &engine = c.getExecutionEngine();

        StructType *s = ModelDataIRBuilder::getStructType(c.getModule());

        EvalInitialConditionsCodeGen iv(c);

        EvalInitialConditionsCodeGen::FunctionPtr pfunc;

        //pfunc = iv.createFunction();

        pfunc(md, 0);

        Log(lInfo) << md;

        cout << "done with " << modelFileName << endl;

    }
    catch(std::exception &e)
    {
        cout << "Failure in " << modelFileName.c_str() << ", " << e.what() << "\n";
    }

    delete doc;

    return true;
}

bool runModelDataAccessorTest(const string& version, int caseNumber)
{
    LLVMModelData *md;
    string modelFileName = getModelFileName(version, caseNumber);

    SBMLDocument *doc = readSBMLFromFile(modelFileName.c_str());

    ModelGeneratorContext c(doc, true);

    md = createModelData(c.getModelDataSymbols(), 0);

    LLVMModelDataIRBuilderTesting builder(c.getModelDataSymbols(), c.getBuilder());

    //builder.test(c.getModule(), c.getBuilder(), c.getExecutionEngine());

    builder.createAccessors(c.getModule());

    ExecutionEngine &engine = c.getExecutionEngine();
    Function *getFunc = engine.FindFunctionNamed("get_size");

    //getFunc->dump();

    // JIT the function, returning a function pointer.


    // Cast it to the right type (takes no arguments, returns a double) so we
    // can call it as a native function.
    int (*pfunc)(LLVMModelData*) = (int (*)(LLVMModelData*))engine.getPointerToFunction(getFunc);

    double value = pfunc(md);

    cout << "get_size returned " << value << "\n";



    vector<string> floatSpeciesIds = c.getModelDataSymbols().getFloatingSpeciesIds();


    for(int i = 0; i < md->numIndCompartments; i++)
    {
        md->compartmentVolumesAlias[i] = 1.5;
    }

    for(int i = 0; i < floatSpeciesIds.size(); i++)
    {
        string getName = "get_floatingspecies_conc_" + floatSpeciesIds[i];
        Function *getFunc = engine.FindFunctionNamed(getName.c_str());

        //getFunc->dump();

        // JIT the function, returning a function pointer.


              // Cast it to the right type (takes no arguments, returns a double) so we
              // can call it as a native function.
        double (*pfunc)(LLVMModelData*) = (double (*)(LLVMModelData*))engine.getPointerToFunction(getFunc);

        double value = pfunc(md);

        cout << getName << " returned " << value << "\n";

        string setName = "set_floatingspecies_conc_" + floatSpeciesIds[i];
        Function *setFunc = engine.FindFunctionNamed(setName.c_str());

        void (*psetfunc)(LLVMModelData*,double) = (void (*)(LLVMModelData*,double))engine.getPointerToFunction(setFunc);

        psetfunc(md, i+1);

    }


    LLVMModelData_free(md);

    delete doc;

    return true;
}


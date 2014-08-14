#ifndef CModelGenerator_H_
#define CModelGenerator_H_
//---------------------------------------------------------------------------
#include "ModelGenerator.h"
#include "rrCompiledModelGenerator.h"
#include "rrModelSharedLibrary.h"
#include "rrCodeBuilder.h"
#include "rrCCompiler.h"

namespace rr
{

using Poco::Mutex;

/**
 * Generate executable SBML models by generating and compiling C
 * source code into shared libraries with an external C compiler.
 */
class RR_DECLSPEC CModelGenerator : public CompiledModelGenerator
{

public:
    CModelGenerator(const string& tempFolder, const string& supportCodeFolder, const string& compiler);
    virtual                            ~CModelGenerator();

    virtual  ExecutableModel            *createModel(const string& sbml, uint options, const std::string& filename);

    virtual bool                        setTemporaryDirectory(const string& path);

    /**
     * Get the location where this model generator creates source file and shared libraries.
     */
    virtual string                      getTemporaryDirectory();

    /**
     * Get the compiler object that the model generator is using to
     * 'compile' sbml.
     */
    virtual                             Compiler* getCompiler();

    /**
     * Set the name of the compiler to use. As this is a C source code compiler, this
     * is the name of the external C compiler, which would typically be 'gcc', 'cc', 'icc', etc...
     */
    virtual                             bool setCompiler(const string& compiler);

    /**
     * These are access in the c_api, should they be??? TODO: fix this.
     */
    string                              getHeaderCode();
    string                              getSourceCode();
    string                              getSourceCodeFileName();

private:
    /**
     * create a C compiled executable model.
     *
     * The caller own this.
     */
    ExecutableModel                     *createModel(const string& sbml, ls::LibStructural *ls,
                                                     bool forceReCompile, bool computeAndAssignConsevationLaws, 
                                                     const std::string& filename);

    CodeBuilder                         mHeader;
    CodeBuilder                         mSource;

    string                              mHeaderCodeFileName;
    string                              mSourceCodeFileName;

    string                              mCurrentSBML;

    // the shared lib thats created.
    // this needs to belong to the model
    // TODO: fix this!
    ModelSharedLibrary                  *mModelLib;

    CCompiler                           mCompiler;

    string                              mTempFileFolder;

    bool generateModelCode(const string& sbml, const string& modelName, bool computeAndAssignConsevationLaws);

    static Mutex                        mCompileMutex;

    bool compileModel();

    bool compileCurrentModel();



    //C Specifics..

    string                              getHeaderCodeFileName();
    bool                                saveSourceCodeToFolder(const string& folder, const string& baseName);
    int                                 getNumberOfFloatingSpecies();


    string                              convertUserFunctionExpression(const string& equation);
    string                              convertCompartmentToC(const string& compartmentName);
    string                              convertSpeciesToBc(const string& speciesName);
    string                              convertSpeciesToY(const string& speciesName);

    /**
     * creates a string of the format "md->c[comp_index]", where
     * comp_index is the index of the named compartment
     */
    string                              convertSymbolToC(const string& compartmentName);
    string                              convertSymbolToGP(const string& parameterName);

    void                                substituteEquation(const string& reactionName, Scanner& s, CodeBuilder& sb);
    void                                substituteWords(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& sb);
    void                                substituteToken(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& sb);
    string                              findSymbol(const string& varName);
    void                                write_getModelNameFunction(CodeBuilder& ignore, CodeBuilder& source);
    void                                writeComputeAllRatesOfChange(CodeBuilder& sb, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0);
    void                                writeComputeConservedTotals(CodeBuilder& sb, const int& numFloatingSpecies, const int& numDependentSpecies);
    void                                writeUpdateDependentSpecies(CodeBuilder& sb, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0);
    void                                writeUserDefinedFunctions(CodeBuilder& sb);
    void                                writeResetEvents(CodeBuilder& sb, const int& numEvents);
    void                                writeSetConcentration(CodeBuilder& sb);
    void                                writeGetConcentration(CodeBuilder& sb);
    void                                writeConvertToAmounts(CodeBuilder& sb);
    void                                writeConvertToConcentrations(CodeBuilder& sb);
    void                                writeProperties(CodeBuilder& sb);
    void                                writeAccessors(CodeBuilder& sb);
    void                                writeOutVariables(CodeBuilder& sb);
    void                                writeClassHeader(CodeBuilder& sb);
    void                                writeTestConstraints(CodeBuilder& sb);
    void                                writeEvalInitialAssignments(CodeBuilder& sb, const int& numReactions);
    int                                 writeComputeRules(CodeBuilder& sb, const int& numReactions);
    void                                writeComputeReactionRates(CodeBuilder& sb, const int& numReactions);
    void                                writeEvalEvents(CodeBuilder& sb, const int& numEvents, const int& numFloatingSpecies);
    void                                writeEvalModel(CodeBuilder& sb, const int& numReactions, const int& numIndependentSpecies, const int& numFloatingSpecies, const int& numOfRules);
    void                                writeEventAssignments(CodeBuilder& sb, const int& numReactions, const int& numEvents);
    void                                writeSetParameterValues(CodeBuilder& sb, const int& numReactions);
    void                                writeSetCompartmentVolumes(CodeBuilder& sb);
    void                                writeSetBoundaryConditions(CodeBuilder& sb);
    void                                writeSetInitialConditions(CodeBuilder& sb, const int& numFloatingSpecies);
    void                                writeInitFunction(CodeBuilder& sbh, CodeBuilder& sbc);
    void                                writeInitModelDataFunction(CodeBuilder& sbh, CodeBuilder& sbc);

    int                                 readFloatingSpecies();
    // int                                 readBoundarySpecies();

    // Generates the Model Code from the SBML string
    // TODO major clean up
    string                              generateModelCode(const string& sbmlStr, const bool& _computeAndAssignConsevationLaws);

    // Generates the Model Code from th e SBML string
    // TODO major clean up also
    string                              generateModelCode(const string& sbmlStr, LibStructural *ls, NOMSupport *nom,
                                                          const bool& _computeAndAssignConsevationLaws = false);

    static bool                         loadSBMLIntoNOM(NOMSupport &nom, const string& sbml, const std::string& filename);

};


}

#endif

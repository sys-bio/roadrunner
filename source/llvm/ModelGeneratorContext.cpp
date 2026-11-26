/*
 * ModelGeneratorContext.cpp
 *
 *  Created on: Jun 18, 2013
 *      Author: andy
 */
#pragma hdrstop

#include "ModelGeneratorContext.h"
#include "rrSparse.h"
#include "LLVMIncludes.h"
#include "ModelDataIRBuilder.h"
#include "LLVMException.h"
#include "SBMLSupportFunctions.h"
#include "AlgebraicRuleConverter.h"
#include "conservation/ConservedMoietyConverter.h"
#include "conservation/ConservationExtension.h"
#include "rrRoadRunnerOptions.h"
#include "rrConfig.h"
#include "rrUtils.h"

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#include <sbml/math/ASTNodeType.h>
#include <sbml/math/FormulaFormatter.h>


#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"


#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

#include <sbml/SBMLReader.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <memory>

#include "ModelResources.h"

#include <sbml/common/libsbml-config-packages.h>

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB

#include <sbml/packages/distrib/common/DistribExtensionTypes.h>

#endif

using namespace llvm;

using namespace libsbml;

using rr::Logger;
using rr::LoadSBMLOptions;

namespace rrllvm {

    static Function *createGlobalMappingFunction(const char *funcName,
                                                 llvm::FunctionType *funcType, Module *module);


    /**
     * returns a VALID sbml document, if the doc has any error,
     * an exception is thrown.
     */
    static SBMLDocument *checkedReadSBMLFromString(const char *xml);

// MSVC 2010 and earlier do not include the hyperbolic functions, define there here
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)
// Note, evidently including the <amp_math.h> causes issues in 2012,
// so it seems to work just defining these here as static.
#if defined(_MSC_VER)

    static double asinh(double value) {
        return log(value + sqrt(value * value + 1.));
    }

    static double acosh(double value) {
        return log(value + sqrt(value * value - 1.));
    }

    double atanh(double value) {
        return (std::log(1.0 + value) - std::log(1.0 - value)) / 2.0;
    }

#endif

    ModelGeneratorContext::ModelGeneratorContext(const libsbml::SBMLDocument* doc_, unsigned options,
                                                 Jit* jitEngine)
            :
            ownedDoc(nullptr),
            doc(nullptr),
            mPiecewiseTriggers(),
            mBooleanLeaves(),
            symbols(nullptr),
            options(options),
            random(nullptr),
            jit(jitEngine) 
    {
        if (useSymbolCache()) {
            rrLog(Logger::LOG_INFORMATION) << "Using LLVM symbol/value cache";
        } else {
            rrLog(Logger::LOG_INFORMATION) << "Not using LLVM symbol/value cache";
        }

        try {

            // The caller's document keeps its algebraic rules, only the copy
            // that gets compiled has them rewritten as assignment rules.
            ownedDoc = convertAlgebraicRules(doc_);
            const libsbml::SBMLDocument* sourceDoc = ownedDoc ? ownedDoc : doc_;

            if (options & LoadSBMLOptions::CONSERVED_MOIETIES) {
                if ((rr::Config::getBool(rr::Config::ROADRUNNER_DISABLE_WARNINGS) &&
                     rr::Config::ROADRUNNER_DISABLE_WARNINGS_CONSERVED_MOIETY) == 0) {
                    rrLog(Logger::LOG_NOTICE) << "performing conserved moiety conversion";
                }

                // check if already conserved doc
                if (rr::conservation::ConservationExtension::isConservedMoietyDocument(sourceDoc)) {
                    doc = sourceDoc;
                } else {
                    moietyConverter = std::make_unique<rr::conservation::ConservedMoietyConverter>();

                    if (moietyConverter->setDocument(sourceDoc) != LIBSBML_OPERATION_SUCCESS) {
                        throw_llvm_exception("error setting conserved moiety converter document");
                    }

                    if (moietyConverter->convert() != LIBSBML_OPERATION_SUCCESS) {
                        throw_llvm_exception("error converting document to conserved moieties");
                    }

                    doc = moietyConverter->getDocument();

                    if (rr::Logger::getLevel() <= rr::Logger::LOG_INFORMATION) {
                        SBMLWriter sw;
                        char *convertedStr = sw.writeToString(doc);

                        rrLog(Logger::LOG_INFORMATION)
                            << "***************** Conserved Moiety Converted Document ***************";
                        rrLog(Logger::LOG_INFORMATION) << convertedStr;
                        rrLog(Logger::LOG_INFORMATION)
                            << "*********************************************************************";

                        free(convertedStr);
                    }
                }
            } else {
                doc = sourceDoc;
            }

            model = doc->getModel();

            addAllBooleanLeaves(model);

            addAllPiecewiseTriggers(model);

            symbols = new LLVMModelDataSymbols(doc->getModel(), options);

            modelSymbols = std::make_unique<LLVMModelSymbols>(getModel(), *symbols);

            // initialize LLVM
            // Note - this initialization is duplicated but some tests (the CAPIModelEditingTests) break
            // without it.
            InitializeNativeTarget();
            InitializeNativeTargetAsmPrinter();
            InitializeNativeTargetAsmParser();

            ModelDataIRBuilder::createModelDataStructType(jit->getModuleNonOwning(), nullptr, *symbols);


            // check if doc has distrib package
            // Random adds mappings, need call after llvm objs created
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
            const DistribSBMLDocumentPlugin *distrib =
                    static_cast<const DistribSBMLDocumentPlugin *>(
                            doc->getPlugin("distrib"));
            if (distrib) {
                random = new Random(*this);
            }
#endif

        }
        catch (const std::exception &) {
            // we might have allocated memory in a failed ctor,
            // clean it up here.
            // destructors are not called on *this* class when exception is raised
            // in the ctor.
            cleanup();
            throw;
        }
    }


    static SBMLDocument *createEmptyDocument() {
        SBMLDocument *doc = new SBMLDocument();
        doc->createModel("");
        return doc;
    }

    ModelGeneratorContext::ModelGeneratorContext() 
        : ownedDoc(createEmptyDocument())
        , doc(ownedDoc)
        , mPiecewiseTriggers()
        , mBooleanLeaves()
        , symbols(new LLVMModelDataSymbols(doc->getModel(), 0))
        , modelSymbols(new LLVMModelSymbols(doc->getModel(), *symbols))
        , model(doc->getModel())
        , random(NULL)
        , options(0)
        , jit(nullptr)
        , moietyConverter(nullptr)
    {
        // initialize LLVM
        // TODO check result
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        InitializeNativeTargetAsmParser();
    }

    Random *ModelGeneratorContext::getRandom() const {
        return random;
    }

    void ModelGeneratorContext::addAllPiecewiseTriggers(const libsbml::Model* model)
    {
        clearPiecewiseTriggers();

        //Only worry about piecewise functions in continuous contexts:  initial assignments and event assignments are fine, because we don't need to stop simulation to check things; the assignment is already happening when simulation is stopped.
        for (size_t r = 0; r < model->getNumRules(); r++) {
            //We don't care whether the rule is assignment, rate, or even algebraic; we're just looking for piecewise functions.
            const libsbml::Rule* rule = model->getRule(r);
            addPiecewiseTriggersFrom(model, rule->getMath(), NULL);
        }
        for (size_t r = 0; r < model->getNumReactions(); r++) {
            const libsbml::KineticLaw* kl = model->getReaction(r)->getKineticLaw();
            if (kl != NULL) {
                addPiecewiseTriggersFrom(model, kl->getMath(), kl);
            }
        }
        //Event triggers themselves might have piecewise arguments in them:
        for (size_t e = 0; e < model->getNumEvents(); e++) {
            const libsbml::Event* event = model->getEvent(e);
            if (event->isSetTrigger()) {
                addPiecewiseTriggersFrom(model, event->getTrigger()->getMath(), NULL);
            }
        }
    }

    // Replaces each name in the tree that is a key of 'bindings' with a copy of its bound node.  The
    // replacement is simultaneous: names inside the copied nodes are not substituted again.
    static void substituteBoundVariables(libsbml::ASTNode* node, const std::map<std::string, const libsbml::ASTNode*>& bindings)
    {
        for (unsigned int c = 0; c < node->getNumChildren(); c++) {
            libsbml::ASTNode* child = node->getChild(c);
            std::map<std::string, const libsbml::ASTNode*>::const_iterator found = bindings.end();
            if (child->getType() == libsbml::AST_NAME) {
                found = bindings.find(child->getName());
            }
            if (found != bindings.end()) {
                node->replaceChild(c, found->second->deepCopy(), true);
            }
            else {
                substituteBoundVariables(child, bindings);
            }
        }
    }

    // A copy of the body of the function with the arguments of the call in place of its bound variables, or null if the call does not match the function.
    static libsbml::ASTNode* expandFunctionCall(const libsbml::FunctionDefinition* function, const libsbml::ASTNode* call)
    {
        const libsbml::ASTNode* lambda = function->getMath();
        const unsigned int numArgs = call->getNumChildren();
        if (lambda == NULL || !lambda->isLambda() || lambda->getNumChildren() != numArgs + 1) {
            return NULL;
        }
        std::map<std::string, const libsbml::ASTNode*> bindings;
        for (unsigned int a = 0; a < numArgs; a++) {
            bindings[lambda->getChild(a)->getName()] = call->getChild(a);
        }
        libsbml::ASTNode* body = lambda->getChild(numArgs)->deepCopy();
        if (body->getType() == libsbml::AST_NAME) {
            std::map<std::string, const libsbml::ASTNode*>::const_iterator found = bindings.find(body->getName());
            if (found != bindings.end()) {
                libsbml::ASTNode* argument = found->second->deepCopy();
                delete body;
                return argument;
            }
        }
        substituteBoundVariables(body, bindings);
        return body;
    }

    // A copy of the math with the calls of user functions expanded, nested calls included, or null if a call can't be expanded.
    static libsbml::ASTNode* expandUserFunctionCalls(const libsbml::Model* model, const libsbml::ASTNode* node, unsigned int depth)
    {
        const unsigned int maxFunctionDepth = 64;
        if (node->getType() == libsbml::AST_FUNCTION && node->getName() != NULL) {
            const libsbml::FunctionDefinition* function = model->getFunctionDefinition(node->getName());
            if (function != NULL) {
                libsbml::ASTNode* body = depth < maxFunctionDepth ? expandFunctionCall(function, node) : NULL;
                if (body == NULL) {
                    return NULL;
                }
                libsbml::ASTNode* expanded = expandUserFunctionCalls(model, body, depth + 1);
                delete body;
                return expanded;
            }
        }
        libsbml::ASTNode* copy = node->deepCopy();
        for (unsigned int c = 0; c < node->getNumChildren(); c++) {
            libsbml::ASTNode* child = expandUserFunctionCalls(model, node->getChild(c), depth);
            if (child == NULL) {
                delete copy;
                return NULL;
            }
            copy->replaceChild(c, child, true);
        }
        return copy;
    }

    void ModelGeneratorContext::addPiecewiseTriggersFrom(const libsbml::Model* model, const libsbml::ASTNode* math, const libsbml::KineticLaw* kl)
    {
        if (math == NULL) {
            return;
        }
        // The calls of user functions are expanded, since the function body is where the piecewise functions are.
        libsbml::ASTNode* expanded = expandUserFunctionCalls(model, math, 0);
        if (expanded == NULL) {
            rrLog(Logger::LOG_WARNING) << "Unable to expand a call of a user function when looking for piecewise functions.  Any transitions in those piecewise functions may not be noticed by the simulator.";
            return;
        }
        addPiecewiseBranchesFrom(expanded, kl);
        delete expanded;
    }

    void ModelGeneratorContext::addPiecewiseBranchesFrom(const libsbml::ASTNode* node, const libsbml::KineticLaw* kl)
    {
        if (node->getType() == libsbml::AST_FUNCTION_PIECEWISE) {
            // The format of the piecewise function is (val1, cond1, val2, cond2, ..., val_otherwise).  They're checked sequentially, so if cond1 is true, val1 is always returned, and if cond1 is false but cond2 is true, val2 is returned, etc.
            // The branch that is taken changes only where a comparison inside a condition changes, so those comparisons are roots.  The trigger of the piecewise function is the index of the branch that it takes,
            // piecewise(0, cond1, 1, cond2, ..., n), which the integrator compares before and after a root.  None of the values need to be checked, just the conditions.
            libsbml::ASTNode* branch = new libsbml::ASTNode(libsbml::AST_FUNCTION_PIECEWISE);
            unsigned int numConditions = 0;
            size_t numLeaves = 0;
            for (size_t c = 1; c < node->getNumChildren(); c += 2) {
                libsbml::ASTNode* index = new libsbml::ASTNode(libsbml::AST_INTEGER);
                index->setValue(static_cast<long>(numConditions++));
                branch->addChild(index);
                branch->addChild(node->getChild(c)->deepCopy());

                std::vector<const libsbml::ASTNode*> leaves;
                addBooleanLeavesFrom(node->getChild(c), leaves);
                for (size_t l = 0; l < leaves.size(); l++) {
                    addBooleanLeaf(leaves[l], kl);
                }
                numLeaves += leaves.size();
            }
            // Conditions that have no comparison never change.
            if (numLeaves > 0) {
                libsbml::ASTNode* otherwise = new libsbml::ASTNode(libsbml::AST_INTEGER);
                otherwise->setValue(static_cast<long>(numConditions));
                branch->addChild(otherwise);
                mPiecewiseTriggers.push_back(branch);
                mPiecewiseTriggerKineticLaws.push_back(kl);
            }
            else {
                delete branch;
            }
        }

        //Not 'else', since a piecewise function can have piecewise children.
        for (size_t c = 0; c < node->getNumChildren(); c++) {
            addPiecewiseBranchesFrom(node->getChild(c), kl);
        }
    }

    void ModelGeneratorContext::clearPiecewiseTriggers()
    {
        for (size_t pt = 0; pt < mPiecewiseTriggers.size(); pt++) {
            delete mPiecewiseTriggers[pt];
        }
        mPiecewiseTriggers.clear();
        mPiecewiseTriggerKineticLaws.clear();
    }

    static bool isRelationalNode(const libsbml::ASTNode* node)
    {
        switch (node->getType()) {
        case libsbml::AST_RELATIONAL_EQ:
        case libsbml::AST_RELATIONAL_GEQ:
        case libsbml::AST_RELATIONAL_GT:
        case libsbml::AST_RELATIONAL_LEQ:
        case libsbml::AST_RELATIONAL_LT:
        case libsbml::AST_RELATIONAL_NEQ:
            return true;
        default:
            return false;
        }
    }

    void ModelGeneratorContext::addBooleanLeavesFrom(const libsbml::ASTNode* node, std::vector<const libsbml::ASTNode*>& leaves)
    {
        if (node == NULL) {
            return;
        }
        if (isRelationalNode(node)) {
            leaves.push_back(node);
            return;
        }
        for (size_t c = 0; c < node->getNumChildren(); c++) {
            addBooleanLeavesFrom(node->getChild(c), leaves);
        }
    }

    void ModelGeneratorContext::addBooleanLeaf(const libsbml::ASTNode* leaf, const libsbml::KineticLaw* kl)
    {
        char* formula = SBML_formulaToL3String(leaf);
        std::string key = formula ? formula : "";
        free(formula);
        if (key.empty()) {
            key = "#" + std::to_string(mBooleanLeaves.size());
        }
        if (mBooleanLeafKeys.insert(std::make_pair(kl, key)).second) {
            mBooleanLeaves.push_back(leaf->deepCopy());
            mBooleanLeafKineticLaws.push_back(kl);
        }
    }

    void ModelGeneratorContext::addAllBooleanLeaves(const libsbml::Model* model)
    {
        clearBooleanLeaves();

        for (size_t e = 0; e < model->getNumEvents(); e++) {
            const libsbml::Event* event = model->getEvent(e);
            if (!event->isSetTrigger() || !event->getTrigger()->isSetMath()) {
                continue;
            }
            libsbml::ASTNode* trigger = expandUserFunctionCalls(model, event->getTrigger()->getMath(), 0);
            if (trigger == NULL) {
                rrLog(Logger::LOG_WARNING) << "Unable to expand a call of a user function in the trigger of event '" << event->getId() << "'.  A trigger that is true only within one step of the integrator may not be noticed by the simulator.";
                continue;
            }
            std::vector<const libsbml::ASTNode*> leaves;
            addBooleanLeavesFrom(trigger, leaves);

            std::set<std::string> distinct;
            for (size_t l = 0; l < leaves.size(); l++) {
                char* formula = SBML_formulaToL3String(leaves[l]);
                std::string key = formula ? formula : "";
                free(formula);
                if (key.empty()) {
                    key = "#" + std::to_string(l);
                }
                distinct.insert(key);
            }

            // A trigger with a single comparison already has its own root.
            if (distinct.size() >= 2) {
                for (size_t l = 0; l < leaves.size(); l++) {
                    addBooleanLeaf(leaves[l], NULL);
                }
            }
            delete trigger;
        }
    }

    void ModelGeneratorContext::clearBooleanLeaves()
    {
        for (size_t lt = 0; lt < mBooleanLeaves.size(); lt++) {
            delete mBooleanLeaves[lt];
        }
        mBooleanLeaves.clear();
        mBooleanLeafKineticLaws.clear();
        mBooleanLeafKeys.clear();
    }

    void ModelGeneratorContext::cleanup() {
        delete ownedDoc;
        ownedDoc = 0;
        clearPiecewiseTriggers();
        clearBooleanLeaves();
        delete symbols;
    }

    ModelGeneratorContext::~ModelGeneratorContext() {
        cleanup();
    }

    const LLVMModelDataSymbols &ModelGeneratorContext::getModelDataSymbols() const {
        return *symbols;
    }

    const libsbml::SBMLDocument *ModelGeneratorContext::getDocument() const {
        return doc;
    }

    Jit *ModelGeneratorContext::getJitNonOwning() const {
        return jit.get();
    }

    const libsbml::Model *ModelGeneratorContext::getModel() const {
        return doc->getModel();
    }

    const std::vector<libsbml::ASTNode*>* ModelGeneratorContext::getPiecewiseTriggers() const
    {
        return &mPiecewiseTriggers;
    }

    const libsbml::KineticLaw* ModelGeneratorContext::getPiecewiseTriggerKineticLaw(size_t index) const
    {
        return mPiecewiseTriggerKineticLaws[index];
    }

    size_t ModelGeneratorContext::getNumPiecewiseTriggers() const
    {
        return mPiecewiseTriggers.size();
    }

    const std::vector<libsbml::ASTNode*>* ModelGeneratorContext::getBooleanLeaves() const
    {
        return &mBooleanLeaves;
    }

    const libsbml::KineticLaw* ModelGeneratorContext::getBooleanLeafKineticLaw(size_t index) const
    {
        return mBooleanLeafKineticLaws[index];
    }

    size_t ModelGeneratorContext::getNumBooleanLeaves() const
    {
        return mBooleanLeaves.size();
    }

    void ModelGeneratorContext::transferObjectsToResources(std::shared_ptr<rrllvm::ModelResources> modelResources) {
        modelResources->symbols = symbols;
        symbols = nullptr;

        modelResources->jit = std::move(jit);
        jit = nullptr;

        modelResources->random = random;
        random = nullptr;
    }

    const LLVMModelSymbols &ModelGeneratorContext::getModelSymbols() const {
        return *modelSymbols;
    }

    bool ModelGeneratorContext::getConservedMoietyAnalysis() const {
        return (options & LoadSBMLOptions::CONSERVED_MOIETIES) != 0;
    }

    bool ModelGeneratorContext::useSymbolCache() const {
        return (options & LoadSBMLOptions::LLVM_SYMBOL_CACHE) != 0;
    }

    static SBMLDocument *checkedReadSBMLFromString(const char *xml) {
        SBMLDocument *doc = readSBMLFromString(xml);

        if (doc) {
            if (doc->getModel() == 0) {
                // fatal error
                SBMLErrorLog *log = doc->getErrorLog();
                std::string errors = log ? log->toString() : " NULL SBML Error Log";
                delete doc;
                throw_llvm_exception("Fatal SBML error, no model, errors in sbml document: " + errors);
            } else if (doc->getNumErrors() > 0) {
                SBMLErrorLog *log = doc->getErrorLog();
                std::string errors = log ? log->toString() : " NULL SBML Error Log";
                rrLog(rr::Logger::LOG_WARNING) << "Warning, errors found in sbml document: " + errors;
            }
        } else {
            delete doc;
            throw_llvm_exception("readSBMLFromString returned NULL, no further information available");
        }
        return doc;
    }


} /* namespace rr */



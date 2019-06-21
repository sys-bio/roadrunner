/*
 * ModelGeneratorContext.h
 *
 *  Created on: Jun 18, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef ModelGeneratorContext_H_
#define ModelGeneratorContext_H_

#include "LLVMIncludes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "LLVMModelDataSymbols.h"
#include "LLVMModelSymbols.h"
#include "Random.h"

#include <sbml/Model.h>
#include <sbml/SBMLDocument.h>
#include <string>

namespace libsbml {
class SBMLDocument;
class Model;
}

namespace rr { namespace conservation {
class ConservedMoietyConverter;
}}

namespace rrllvm
{

/**
 * All LLVM code generating objects basically need at a minimum three things
 * to operate:
 *
 * 1: sbml Model - what to process
 * 2: llvm Context - all llvm functions need this, manages llvm memory
 * 3: llvm Module - where the generated code will go
 *
 *
 * Code generators also need to know the symbols that are in the sbml doc.
 * These *could* be re-determined in each code gen, but wastefull. The
 * symbols can be thought of as the first pass of the compiler, so all symbols
 * are determined ahead of time before any code generation.
 *
 * They also make use of an llvm IRBuilder, one of these *could* be allocated
 * in each code gen object, but that would be wasteful, so we use a single one
 * per sbml compilation, and each code gen just resets the builder insert point,
 * that is what is was designed to do.
 *
 * So, that leaves us with 5 object that are needed by code generators.
 * Should all code gens have all 5 args passed to thier ctors? There is much
 * talk about how context objects are evil, however this one is NOT inteneded to
 * grow, only store the absolute minimun information that all code generators
 * need.
 *
 * This class basically functions as a set of ivars if all code generators
 * were methods on a single code gen class.
 *
 * TODO doc wise discuss the problems with context objects, and why this one
 * address all the concerns.
 *
 * TODO document memory managment, engine owns model, ...
 */
class ModelGeneratorContext
{
public:
    /**
     * load the sbml document from a string.
     */
    ModelGeneratorContext(std::string const &sbml, unsigned loadSBMLOptions);

    /**
     * attach to an existing sbml document, we borrow a reference to this
     * doc and DO NOT take ownership of it.
     */
    ModelGeneratorContext(libsbml::SBMLDocument const *doc,
            unsigned loadSBMLOptions);

    /**
     * does not attach to any sbml doc,
     *
     * useful for testing out LLVM functionality.
     */
    ModelGeneratorContext();

    ~ModelGeneratorContext();

    const LLVMModelDataSymbols &getModelDataSymbols() const;

    const LLVMModelSymbols &getModelSymbols() const;

    const libsbml::SBMLDocument *getDocument() const;

    const libsbml::Model *getModel() const;


    llvm::LLVMContext &getContext() const;

    llvm::ExecutionEngine &getExecutionEngine() const;

    /**
     * nearly all llvm functions expect a pointer to module, so we define this
     * as a pointer type instead of reference, even though we gaurantee this to
     * be non-null
     */
    llvm::Module *getModule() const;

    /**
     * if optimization is enabled, this gets the function pass
     * manager loaded with all the requested optimizers.
     * NULL if no optimization is specified.
     */
    llvm::legacy::FunctionPassManager *getFunctionPassManager() const;

    llvm::IRBuilder<> &getBuilder() const;

    /**
     * A lot can go wrong in the process of generating a model from  an sbml doc.
     * This class is intended to be stack allocated, so when any exception is
     * thrown in the course of model generation, this class will clean
     * up all the contexts and execution engines and so forth.
     *
     * However, when a model is successfully generated, we need a way to give
     * it the exec engine, and whatever other bits it requires.
     *
     * So, this method exists so that the generated model can steal all the
     * objects it needs from us, these object are transfered to the model,
     * and our pointers to them are cleared.
     *
     * Monkey steals the peach -- A martial arts technique mastered by
     * Michael Wu which is in effect, the act of ripping someone's bollocks off.
     */
    void stealThePeach(const LLVMModelDataSymbols **sym,
             llvm::LLVMContext **ctx,  llvm::ExecutionEngine **eng,
            const Random **random, const std::string **errStr, std::unique_ptr<llvm::Module> &mod);


    bool getConservedMoietyAnalysis() const;

    bool useSymbolCache() const;

    unsigned getOptions() const
    {
        return options;
    }

    /**
     * get a pointer to the random object.
     *
     * The random object exists if the document has the distrib package,
     * is null otherwise.
     */
    Random* getRandom() const;

private:

    /**
     * set the execution engine's global mappings to the rr functions
     * that are accessible from the LLVM generated code.
     */
	void addGlobalMapping(const llvm::GlobalValue * gv, void * addr);
	void addGlobalMappings();

    /**
     * these point to the same location, ownedDoc is set if we create the doc,
     * otherwise its 0, meaning we're borrowign the the doc.
     */
    libsbml::SBMLDocument *ownedDoc;

    /**
     * allways references the sbml doc.
     */
    const libsbml::SBMLDocument *doc;

    LLVMModelDataSymbols *symbols;

    /**
     * make sure this is listed AFTER the doc and model, so it get
     * initialized after the previous two are initialized.
     */
    LLVMModelSymbols *modelSymbols;

    std::string *errString;

    llvm::LLVMContext *context;
    llvm::ExecutionEngine *executionEngine;
    std::unique_ptr<llvm::Module> module_uniq;
public:
	llvm::Module* module;
private:
    llvm::IRBuilder<> *builder;

    llvm::legacy::FunctionPassManager *functionPassManager;

    /**
     * As the model is being generated, various distributions may be created
     * which are added to the random object.
     *
     * Ownership of the random object is then transfered to the ExecutableModel
     * which owns up untill the ExecutableModel is destroyed.
     *
     * The logic here is that many distributions used the normal_distribution
     * object which maintains a state and has an expensive startup cost. So,
     * the distributions are created whilst the model is being generated,
     * and then the distribution calls just end up calling an existing
     * distribution object inside this class.
     *
     * A pointer to this object is stored in the ModelData struct, and when
     * the generated sbml code calls a stochastic dist function, it passes
     * in a pionter to this object so the func can look up the distribution.
     *
     * The alternative to this approach would have been to simply have global
     * distributions and a global RNG. This however would use more memory and
     * would not be thread safe, and the RNG would have to be shared amoungst
     * multiple models, and would not be be able to create a repeatable stream
     * of random numbers.
     */
    Random *random;

    unsigned options;

    /**
     * the moiety converter, for the time being owns the
     * converted document.
     */
    rr::conservation::ConservedMoietyConverter *moietyConverter;

    void initFunctionPassManager();

    /**
     * free any memory this class allocated.
     */
    void cleanup();
};


LLVMModelData *createModelData(const rrllvm::LLVMModelDataSymbols &symbols,
        const Random* random);


} /* namespace rr */
#endif /* ModelGeneratorContext_H_ */

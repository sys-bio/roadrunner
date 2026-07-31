/*
 * EvalInitialConditionsCodeGen.cpp
 *
 *  Created on: Jun 29, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EvalInitialConditionsCodeGen.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "SBMLInitialValueSymbolResolver.h"
#include "ModelInitialValueSymbolResolver.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"
#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <Poco/Logger.h>


using namespace libsbml;
using namespace llvm;


using rr::Logger;
using rr::getLogger;
using rr::LoadSBMLOptions;


namespace rrllvm
{

const char* EvalInitialConditionsCodeGen::FunctionName = "evalInitialConditions";

EvalInitialConditionsCodeGen::EvalInitialConditionsCodeGen(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<EvalInitialConditions_FunctionPtr>(mgc)
{
}

EvalInitialConditionsCodeGen::~EvalInitialConditionsCodeGen()
{
}

Value* EvalInitialConditionsCodeGen::codeGen()
{
    // make the set init value function
    llvm::Type *argTypes[] = {
        llvm::PointerType::get(ModelDataIRBuilder::getStructType(this->module), 0),
        llvm::Type::getInt32Ty(this->context)
    };

    const char *argNames[] = {
        "modelData", "flags"
    };

    llvm::Value *args[] = {0, 0};

    //The side effects of creating this codeGenHeader are required for functionality, 
    // even though we don't use the element produced.
    codeGenHeader(FunctionName,
            llvm::Type::getVoidTy(context),
            argTypes, argNames, args);

    Value* modelData = args[0];


    if (Logger::LOG_DEBUG <= rr::Logger::getLevel())
    {
        rrLog(Logger::LOG_DEBUG) << "boundarySpecies: \n";
        const SymbolForest::Map&  initValues = modelSymbols.getInitialValues().boundarySpecies;

        for (SymbolForest::Map::const_iterator i = initValues.begin();
                i != initValues.end(); i++)
        {
            char* formula = SBML_formulaToL3String(i->second);
            rrLog(Logger::LOG_DEBUG) << "\t" << i->first << ": " << formula << "\n";
            free(formula);
        }
    }

    // read symbols from the orginal sbml
    SBMLInitialValueSymbolResolver initialValueResolver(modelData, modelGenContext);

    // store values in the model state
    ModelDataStoreSymbolResolver modelDataResolver(modelData, model,
            modelSymbols, dataSymbols, builder, initialValueResolver);

    // generate model code for both floating and boundary species
    codeGenSpecies(modelDataResolver, initialValueResolver);

    // read init values from sbml and store in model data
    codeGenGlobalParameters(modelDataResolver, initialValueResolver);

    // read values from the model data state std::vector
    ModelDataLoadSymbolResolver modelValueResolver(modelData, this->modelGenContext);


    // initializes the values stored in the model
    // to the values specified in the sbml.

    // always generate code for compartment init values so they are initialized the
    // in the exe model ctor. compartments are not usually reset.
    codeGenCompartments(modelDataResolver, initialValueResolver);

    // store stoich in the sparse matrix structure.
    codeGenStoichiometry(modelData, modelDataResolver, initialValueResolver);

    // at this point, the model data state variables are all initialized
    // from the original sbml values.

    // generates code to set the *initial* values in the model to
    // the values specified in the sbml.
    // at this point, all the model vars have been set,
    // so we can just copy them to the init value locations.
    if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
    {
        // store symbols in the model data init var locations.
        ModelInitialValueStoreSymbolResolver initValueStoreResolver(modelData, model,
                        modelSymbols, dataSymbols, builder, initialValueResolver);

        codeGenInitSpecies(initValueStoreResolver, modelValueResolver);

        codeGenInitCompartments(initValueStoreResolver, modelValueResolver);

        codeGenInitGlobalParameters(initValueStoreResolver, modelValueResolver);

        codeGenInitStoichiometry(modelData);
    }

    builder.CreateRetVoid();

    return verifyFunction();
}

void EvalInitialConditionsCodeGen::codeGenSpecies(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    {
        std::vector<std::string> floatingSpecies = dataSymbols.getFloatingSpeciesIds();

        for (std::vector<std::string>::const_iterator i = floatingSpecies.begin();
                i != floatingSpecies.end(); i++)
        {
            const std::string& id = *i;

            if (!dataSymbols.hasAssignmentRule(id))
            {
                modelDataResolver.storeSymbolValue(id,
                        initialValueResolver.loadSymbolValue(id));
            }
        }
    }


    {
        std::vector<std::string> boundarySpecies = dataSymbols.getBoundarySpeciesIds();

        for (std::vector<std::string>::const_iterator i = boundarySpecies.begin();
                i != boundarySpecies.end(); i++)
        {
            const std::string& id = *i;

            if (!dataSymbols.hasAssignmentRule(id))
            {
                modelDataResolver.storeSymbolValue(id,
                        initialValueResolver.loadSymbolValue(id));
            }
        }
    }
}


void EvalInitialConditionsCodeGen::codeGenStoichiometry(
        llvm::Value *modelData, ModelDataStoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    ModelDataIRBuilder modelDataBuilder(modelData, dataSymbols,
                builder);
    ASTNodeCodeGen astCodeGen(builder, initialValueResolver, modelGenContext, modelData);

    rrLog(Logger::LOG_DEBUG) << "reactions: ";
    std::vector<std::string> ids = dataSymbols.getReactionIds();
    for (int i = 0; i < ids.size(); i++)
    {
        rrLog(Logger::LOG_DEBUG) << ids[i] << ", ";
    }
    rrLog(Logger::LOG_DEBUG) << "\n";

    Value *stoichEP = modelDataBuilder.createGEP(Stoichiometry);
    Value *stoich = builder.CreateLoad(stoichEP->getType()->getPointerElementType(), stoichEP, "stoichiometry");

    std::list<LLVMModelDataSymbols::SpeciesReferenceInfo> stoichEntries =
            dataSymbols.getStoichiometryList();

    for (std::list<LLVMModelDataSymbols::SpeciesReferenceInfo>::iterator i =
            stoichEntries.begin(); i != stoichEntries.end(); i++)
    {
        LLVMModelDataSymbols::SpeciesReferenceInfo nz = *i;
        const ASTNode *node = modelSymbols.createStoichiometryNode(nz.row, nz.column);
        char* formula = SBML_formulaToL3String(node);
        rrLog(Logger::LOG_DEBUG) << "\t{" << nz.row << ", " << nz.column << "} : " << formula
                << "\n";
        free(formula);

        // createCSRMatrixSetNZ(llvm::Value *csrPtr, llvm::Value *row,
        // llvm::Value *col, llvm::Value *value, const char* name = 0);

        Value *stoichValue = astCodeGen.codeGenDouble(node);

        delete node;

        // species references may be defined by rate rules, so set the
        // initial value here. In this case, data is duplicated between the
        // rate rules std::vector and the CSR sparse matrix.
        //
        // MultiSpeciesReference-typed cells are excluded here: nz.id/nz.type
        // are the PRIMARY colliding reference's, but stoichValue is the
        // whole cell's summed value, not that one reference's own -- seeding
        // the rate rule from it would be wrong. The per-reference loop below
        // seeds each colliding reference's rate rule from its own value
        // instead.
        if (!nz.id.empty() && nz.type != LLVMModelDataSymbols::SpeciesReferenceType::MultiSpeciesReference
                && dataSymbols.hasRateRule(nz.id))
        {
            // stoichValue is the net, CSR-signed value (negative for a
            // reactant); the rate rule slot holds the reference's own
            // magnitude, so undo that sign flip before storing it there.
            Value *rateRuleSeed = stoichValue;
            if (nz.type == LLVMModelDataSymbols::SpeciesReferenceType::Reactant)
            {
                Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
                rateRuleSeed = builder.CreateFMul(negOne, stoichValue, "unneg_" + nz.id);
            }
            modelDataBuilder.createRateRuleValueStore(nz.id, rateRuleSeed);
        }

        Value *row = ConstantInt::get(Type::getInt32Ty(context), nz.row, true);
        Value *col = ConstantInt::get(Type::getInt32Ty(context), nz.column, true);
        ModelDataIRBuilder::createCSRMatrixSetNZ(builder, stoich, row, col, stoichValue);

    }

    // seed each MultiSpeciesReference-typed named stoichiometry's own
    // independent storage slot with its own value (StoichiometryMath or
    // literal attribute, unsigned, same "raw passthrough" convention as
    // any other named stoichiometry) -- the cell-level loop above only
    // ever sees the shared, summed value, not each reference's own.
    for (size_t i = 0; i < dataSymbols.getMultiSpeciesReferenceSize(); ++i)
    {
        const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                dataSymbols.getMultiSpeciesReferenceInfo(static_cast<int>(i));

        SymbolForest::Map::const_iterator refNode =
                modelSymbols.getInitialValues().speciesReferences.find(info.id);
        assert(refNode != modelSymbols.getInitialValues().speciesReferences.end() &&
                "MultiSpeciesReference id missing from initial values speciesReferences map");

        Value *refValue = astCodeGen.codeGenDouble(refNode->second);

        // refValue is already this reference's own unsigned value (no sign
        // undo needed here, unlike the cell-level loop above).
        if (dataSymbols.hasRateRule(info.id))
        {
            modelDataBuilder.createRateRuleValueStore(info.id, refValue);
        }

        Value *gep = modelDataBuilder.createGEP(MultiSpeciesReferencesAlias,
                static_cast<unsigned>(i), info.id);
        builder.CreateStore(refValue, gep);
    }
}

void EvalInitialConditionsCodeGen::codeGenInitStoichiometry(llvm::Value *modelData)
{
    // mirrors codeGenInitSpecies/Compartments/GlobalParameters: pass 1
    // (codeGenStoichiometry, above) has already computed the true t=0
    // value of every stoichiometry-matrix cell, so just copy it over to
    // the frozen init matrix rather than re-evaluating anything.
    ModelDataIRBuilder modelDataBuilder(modelData, dataSymbols, builder);

    Value *stoichEP = modelDataBuilder.createGEP(Stoichiometry);
    Value *stoich = builder.CreateLoad(stoichEP->getType()->getPointerElementType(), stoichEP, "stoichiometry");

    Value *initStoichEP = modelDataBuilder.createGEP(InitStoichiometry);
    Value *initStoich = builder.CreateLoad(initStoichEP->getType()->getPointerElementType(), initStoichEP, "initStoichiometry");

    std::list<LLVMModelDataSymbols::SpeciesReferenceInfo> stoichEntries =
            dataSymbols.getStoichiometryList();

    for (std::list<LLVMModelDataSymbols::SpeciesReferenceInfo>::iterator i =
            stoichEntries.begin(); i != stoichEntries.end(); i++)
    {
        LLVMModelDataSymbols::SpeciesReferenceInfo nz = *i;

        Value *row = ConstantInt::get(Type::getInt32Ty(context), nz.row, true);
        Value *col = ConstantInt::get(Type::getInt32Ty(context), nz.column, true);

        Value *value = ModelDataIRBuilder::createCSRMatrixGetNZ(builder, stoich, row, col);
        ModelDataIRBuilder::createCSRMatrixSetNZ(builder, initStoich, row, col, value);
    }

    // mirror the same copy for each MultiSpeciesReference's own independent
    // storage slot.
    for (size_t i = 0; i < dataSymbols.getMultiSpeciesReferenceSize(); ++i)
    {
        Value *srcGEP = modelDataBuilder.createGEP(MultiSpeciesReferencesAlias, static_cast<unsigned>(i));
        Value *refValue = builder.CreateLoad(srcGEP->getType()->getPointerElementType(), srcGEP);

        Value *dstGEP = modelDataBuilder.createGEP(MultiSpeciesReferencesInitAlias, static_cast<unsigned>(i));
        builder.CreateStore(refValue, dstGEP);
    }
}

void EvalInitialConditionsCodeGen::codeGenCompartments(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    std::vector<std::string> compartments = dataSymbols.getCompartmentIds();

    for (std::vector<std::string>::const_iterator i = compartments.begin();
            i != compartments.end(); i++)
    {
        const std::string& id = *i;

        if (!dataSymbols.hasAssignmentRule(id))
        {
            modelDataResolver.storeSymbolValue(id,
                    initialValueResolver.loadSymbolValue(id));
        }
    }
}

void EvalInitialConditionsCodeGen::codeGenInitCompartments(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    std::vector<std::string> compartments = dataSymbols.getCompartmentIds();

    for (std::vector<std::string>::const_iterator i = compartments.begin();
            i != compartments.end(); i++)
    {
        const std::string& id = *i;

        if (!dataSymbols.hasAssignmentRule(id) && !dataSymbols.hasInitialAssignmentRule(id))
        {
            modelDataResolver.storeSymbolValue(id,
                    initialValueResolver.loadSymbolValue(id));
        }
    }
}

void EvalInitialConditionsCodeGen::codeGenInitSpecies(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    std::vector<std::string> floatingSpecies = dataSymbols.getFloatingSpeciesIds();

    for (std::vector<std::string>::const_iterator i = floatingSpecies.begin();
        i != floatingSpecies.end(); i++)
    {
        const std::string& id = *i;

        if (dataSymbols.isIndependentInitFloatingSpecies(id))
        {
            modelDataResolver.storeSymbolValue(id,
                initialValueResolver.loadSymbolValue(id));
        }
    }
    std::vector<std::string> boundarySpecies = dataSymbols.getBoundarySpeciesIds();

    for (std::vector<std::string>::const_iterator i = boundarySpecies.begin();
        i != boundarySpecies.end(); i++)
    {
        const std::string& id = *i;

        if (dataSymbols.isIndependentInitBoundarySpecies(id))
        {
            modelDataResolver.storeSymbolValue(id,
                initialValueResolver.loadSymbolValue(id));
        }
    }
}


void EvalInitialConditionsCodeGen::codeGenGlobalParameters(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    std::vector<std::string> globalParameters = dataSymbols.getGlobalParameterIds();

    for (std::vector<std::string>::const_iterator i = globalParameters.begin();
            i != globalParameters.end(); i++)
    {
        const std::string& id = *i;

        if (!dataSymbols.hasAssignmentRule(id))
        {
            modelDataResolver.storeSymbolValue(id,
                    initialValueResolver.loadSymbolValue(id));
        }
    }
}

void EvalInitialConditionsCodeGen::codeGenInitGlobalParameters(
        StoreSymbolResolver& modelDataResolver,
        LoadSymbolResolver& initialValueResolver)
{
    std::vector<std::string> parameters = dataSymbols.getGlobalParameterIds();

    for (std::vector<std::string>::const_iterator i = parameters.begin();
            i != parameters.end(); i++)
    {
        const std::string& id = *i;

        if (!dataSymbols.hasAssignmentRule(id) && !dataSymbols.hasInitialAssignmentRule(id))
        {
            modelDataResolver.storeSymbolValue(id,
                    initialValueResolver.loadSymbolValue(id));
        }
    }
}

} /* namespace rr */

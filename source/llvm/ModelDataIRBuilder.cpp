/*
 * ModelDataIRBuilder.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ModelDataIRBuilder.h"
#include "LLVMException.h"
#include "rrOSSpecifics.h"

#include <sbml/Model.h>
#include <sbml/SBMLDocument.h>

#include <string>
#include <vector>
#include <sstream>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif
#include "llvm/ADT/APInt.h"
#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif


using namespace libsbml;
using namespace llvm;


using rr::Logger;
using rr::getLogger;
using rr::csr_matrix;

namespace rrllvm
{

const char* ModelDataIRBuilder::LLVMModelDataName = "rr_LLVMModelData";
const char* ModelDataIRBuilder::csr_matrixName = "rr_csr_matrix";
const char* ModelDataIRBuilder::csr_matrix_set_nzName = "csr_matrix_set_nz";
const char* ModelDataIRBuilder::csr_matrix_get_nzName = "csr_matrix_get_nz";


static bool isAliasOrPointer(ModelDataFields f)
{
    /* Model Data alias are between one of the following values */
    /*
        "StateVector",                          // 18
        "StateVectorRate",                      // 19
        "RateRuleRates",                        // 20
        "FloatingSpeciesAmountRates",           // 21

        "CompartmentVolumesAlias",              // 22
        "InitCompartmentVolumesAlias",          // 23
        "InitFloatingSpeciesAmountsAlias",      // 24
        "BoundarySpeciesAmountsAlias",          // 25
        "InitBoundarySpeciesAmountsAlias",      // 26
        "GlobalParametersAlias",                // 27
        "InitGlobalParametersAlias",            // 28
        "ReactionRatesAlias",                   // 29

        "RateRuleValuesAlias",                  // 30
        "FloatingSpeciesAmountsAlias",          // 31
     */
    return f >= StateVector && f <= FloatingSpeciesAmountsAlias;
}

static bool isArray(ModelDataFields f)
{
    /* arrays are the last elements in the model data struct */
    /*
        "CompartmentVolumes",                   // 32
        "InitCompartmentVolumes",               // 33
        "InitFloatingSpeciesAmounts",           // 34
        "BoundarySpeciesAmounts",               // 35
        "InitBoundarySpeciesAmounts",           // 36
        "GlobalParameters",                     // 37
        "InitGlobalParameters",                 // 38
        "ReactionRates",                        // 39
        "NotSafe_RateRuleValues",               // 40
        "NotSafe_FloatingSpeciesAmounts"        // 41
     */

    return f >= CompartmentVolumes && f <= NotSafe_FloatingSpeciesAmounts;
}


ModelDataIRBuilder::ModelDataIRBuilder(Value *modelData,
        const LLVMModelDataSymbols &symbols,
        IRBuilder<> &b) :
        modelData(modelData),
        builder(b),
        symbols(symbols)

{
    validateStruct(modelData, __FUNC__);
}


llvm::Value* ModelDataIRBuilder::createGlobalParamGEP(const std::string& id)
{
    int index = symbols.getGlobalParameterIndex(id);
    assert(index < symbols.getIndependentGlobalParameterSize());
    assert(index >= 0);
    return createGEP(GlobalParameters, index);
}


llvm::Value* ModelDataIRBuilder::createGEP(ModelDataFields field)
{
    const char* fieldName = LLVMModelDataSymbols::getFieldName(field);
	// FIXME:
	// The interface for this function was changed, and I'm unsure how to get the correct type, so I just made it double
	// Okay, now I think I understand how to get the actual type
    return builder.CreateStructGEP(cast<PointerType>(modelData->getType()->getScalarType())->getElementType(), modelData, (unsigned)field, Twine(fieldName) + Twine("_gep"));
}


llvm::Value* ModelDataIRBuilder::createGEP(ModelDataFields field,
        unsigned index, const Twine& name)
{
    if (isAliasOrPointer(field))
    {
        Twine fieldName = LLVMModelDataSymbols::getFieldName(field);
        Value *fieldGEP = createGEP(field);
        Value *load = builder.CreateLoad(fieldGEP, fieldName + "_load");
        return builder.CreateConstGEP1_32(load, index, name + "_gep");
    }
    else
    {
        assert(isArray(field) && "field is not alias or pointer, so it must be an array type");

        LLVMContext &context = builder.getContext();

        Value *idxs[] = {
            ConstantInt::get(Type::getInt32Ty(context), 0),
            ConstantInt::get(Type::getInt32Ty(context), field),
            ConstantInt::get(Type::getInt32Ty(context), index),
        };

        return builder.CreateInBoundsGEP(modelData, idxs, name + "_gep");
    }
}

/**
 * @brief abstract out the llvm::StructType::getTypeByName which
 * has been removed in llvm 12.
 */
llvm::StructType *getTypeByName(llvm::Module *module, std::string name)
{
    llvm::StructType *t;
#if LLVM_VERSION_MAJOR >= 13
    t = llvm::StructType::getTypeByName(module->getContext(), name);
#else
    t = module->getTypeByName(name);
#endif
  return t;
}


llvm::StructType* ModelDataIRBuilder::getCSRSparseStructType(
        llvm::Module* module, llvm::ExecutionEngine* engine)
{
    StructType *structType = getTypeByName(module, csr_matrixName);

    if (!structType)
    {
        LLVMContext &context = module->getContext();

        std::vector<Type*> elements;

        elements.push_back(Type::getInt32Ty(context));                        // int m;
        elements.push_back(Type::getInt32Ty(context));                        // int n;
        elements.push_back(Type::getInt32Ty(context));                        // int nnz;
        elements.push_back(Type::getDoublePtrTy(context));                    // double* values;
        elements.push_back(Type::getInt32PtrTy(context));                     // int* colidx;
        elements.push_back(Type::getInt32PtrTy(context));                     // int* rowptr;

        structType = StructType::create(context, elements, csr_matrixName);

        if (engine)
        {
#if (LLVM_VERSION_MAJOR > 3)
			const DataLayout& dl = engine->getDataLayout();
			uint64_t llvm_size = dl.getTypeStoreSize(structType);
#elif(LLVM_VERSION_MAJOR >= 3) && (LLVM_VERSION_MINOR >= 2)
            const DataLayout *dl = engine->getDataLayout();
            uint64_t llvm_size = dl->getTypeStoreSize(structType);
#else
            const TargetData* td = engine->getTargetData();
            uint64_t llvm_size = td->getTypeStoreSize(structType);
#endif

            if (sizeof(csr_matrix) != llvm_size)
            {
                std::stringstream err;
                err << "llvm " << csr_matrixName << " size " << llvm_size <<
                        " does NOT match C++ sizeof(dcsr_matrix) " <<
                        sizeof(csr_matrix);
                throw LLVMException(err.str(), __FUNC__);
            }
        }
    }
    return structType;
}


llvm::Function* ModelDataIRBuilder::getCSRMatrixSetNZDecl(Module *module)
{
    Function *f = module->getFunction(ModelDataIRBuilder::csr_matrix_set_nzName);

    if (f == 0)
    {
        // bool csr_matrix_set_nz(csr_matrix *mat, int row, int col, double val);
        Type *args[] = {
                getCSRSparseStructType(module)->getPointerTo(),
                Type::getInt32Ty(module->getContext()),
                Type::getInt32Ty(module->getContext()),
                Type::getDoubleTy(module->getContext())
        };
        FunctionType *funcType = FunctionType::get(
                IntegerType::get(module->getContext(), sizeof(bool) * 8), args,
                false);
        f = Function::Create(funcType, Function::ExternalLinkage,
                ModelDataIRBuilder::csr_matrix_set_nzName, module);
    }
    return f;
}

llvm::Function* ModelDataIRBuilder::getCSRMatrixGetNZDecl(Module *module)
{
    Function *f = module->getFunction(
            ModelDataIRBuilder::csr_matrix_get_nzName);

    if (f == 0)
    {
        // double csr_matrix_get_nz(const csr_matrix* mat, int row, int col)
        Type *args[] = {
                getCSRSparseStructType(module)->getPointerTo(),
                Type::getInt32Ty(module->getContext()),
                Type::getInt32Ty(module->getContext())
        };
        FunctionType *funcType = FunctionType::get(
                Type::getDoubleTy(module->getContext()), args, false);
        f = Function::Create(funcType, Function::ExternalLinkage,
                ModelDataIRBuilder::csr_matrix_get_nzName, module);
    }
    return f;
}

llvm::CallInst* ModelDataIRBuilder::createCSRMatrixSetNZ(IRBuilder<> &builder,
        llvm::Value* csrPtr, llvm::Value* row, llvm::Value* col,
        llvm::Value* value, const Twine& name)
{
    Function *func = ModelDataIRBuilder::getCSRMatrixSetNZDecl(getModule(builder, __FUNC__));
    Value *args[] = {csrPtr, row, col, value};
    return builder.CreateCall(func, args, name);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtGEP(
        const std::string& id, const Twine& name)
{
    int index = symbols.getFloatingSpeciesIndex(id);
    assert(index < symbols.getIndependentFloatingSpeciesSize());
    assert(index >= 0);

    return createGEP(FloatingSpeciesAmountsAlias, index,
            name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtLoad(
        const std::string& id, const llvm::Twine& name)
{
    Value *gep = createFloatSpeciesAmtGEP(id, name + "_gep");
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtStore(
        const std::string& id, llvm::Value* value)
{
    Value *gep = createFloatSpeciesAmtGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtRateGEP(
        const std::string& id, const Twine& name)
{
    int index = symbols.getFloatingSpeciesIndex(id);
    assert(index < symbols.getIndependentFloatingSpeciesSize());
    assert(index >= 0);
    return createGEP(FloatingSpeciesAmountRates, index,
            name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtRateLoad(
        const std::string& id, const llvm::Twine& name)
{
    Value *gep = createFloatSpeciesAmtRateGEP(id, name + "_gep");
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createFloatSpeciesAmtRateStore(
        const std::string& id, llvm::Value* value)
{
    Value *gep = createFloatSpeciesAmtRateGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Module* ModelDataIRBuilder::getModule(IRBuilder<> &builder, const char* func)
{
    BasicBlock *bb = 0;
    if((bb = builder.GetInsertBlock()) != 0)
    {
        Function *function = bb->getParent();
        if(function)
        {
            return function->getParent();
        }
    }
    throw LLVMException("could not get module, a BasicBlock is not currently being populated.", func);
}


llvm::CallInst* ModelDataIRBuilder::createCSRMatrixGetNZ(IRBuilder<> &builder,
        llvm::Value* csrPtr, llvm::Value* row, llvm::Value* col,
        const Twine& name)
{
    Function *func = ModelDataIRBuilder::getCSRMatrixGetNZDecl(getModule(builder, __FUNC__));
    Value *args[] = {csrPtr, row, col};
    return builder.CreateCall(func, args, name);
}


llvm::Value* ModelDataIRBuilder::createLoad(ModelDataFields field, unsigned index, const llvm::Twine& name)
{
    Value *gep = this->createGEP(field, index, name);
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createRateRuleValueGEP(const std::string& id,
    const llvm::Twine& name)
{
    int index = symbols.getRateRuleIndex(id);
    assert(index < symbols.getRateRuleSize());
    assert(index >= 0);
    return createGEP(RateRuleValuesAlias, index,
        name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createRateRuleValueLoad(const std::string& id,
        const llvm::Twine& name)
{
    Value *gep = createRateRuleValueGEP(id);
    Twine loadName = (name.isTriviallyEmpty() ? id : name) + "_load";
    return builder.CreateLoad(gep, loadName);
}

llvm::Value* ModelDataIRBuilder::createRateRuleValueStore(const std::string& id,
        llvm::Value* value)
{
    Value *gep = createRateRuleValueGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createRateRuleRateGEP(const std::string& id,
        const llvm::Twine& name)
{
    int index = symbols.getRateRuleIndex(id);
    assert(index < symbols.getRateRuleSize());
    assert(index >= 0);
    return createGEP(RateRuleRates, index,
            name.isTriviallyEmpty() ? id + "_rate" : name);
}

llvm::Value* ModelDataIRBuilder::createRateRuleRateLoad(const std::string& id,
        const llvm::Twine& name)
{
    Value *gep = createRateRuleRateGEP(id);
    Twine loadName = (name.isTriviallyEmpty() ? id : name) + "_load";
    return builder.CreateLoad(gep, loadName);
}

llvm::Value* ModelDataIRBuilder::createRateRuleRateStore(const std::string& id,
        llvm::Value* value)
{
    Value *gep = createRateRuleRateGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createStore(ModelDataFields field,
        unsigned index, llvm::Value* value, const Twine& name)
{
    Value *gep = this->createGEP(field, index, name);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createCompLoad(const std::string& id,
        const llvm::Twine& name)
{
    Value *gep = createCompGEP(id);
    return builder.CreateLoad(gep, name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createCompStore(const std::string& id,
        llvm::Value* value)
{
    Value *gep = createCompGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createCompGEP(const std::string& id)
{
    int index = symbols.getCompartmentIndex(id);
    assert(index < symbols.getIndependentCompartmentSize());
    assert(index >= 0);
    return createGEP(CompartmentVolumes, index, id);
}

llvm::Value* ModelDataIRBuilder::createBoundSpeciesAmtLoad(
        const std::string& id, const llvm::Twine& name)
{
    Value *gep = createBoundSpeciesAmtGEP(id, name + "_gep");
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createBoundSpeciesAmtStore(
        const std::string& id, llvm::Value* value)
{
    Value *gep = createBoundSpeciesAmtGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createBoundSpeciesAmtGEP(
        const std::string& id, const llvm::Twine& name)
{
    int index = symbols.getBoundarySpeciesIndex(id);
    assert(index < symbols.getIndependentBoundarySpeciesSize());
    assert(index >= 0);
    return createGEP(BoundarySpeciesAmounts, index, name);
}

llvm::Value* ModelDataIRBuilder::createGlobalParamLoad(
        const std::string& id, const llvm::Twine& name)
{
    int idx = symbols.getGlobalParameterIndex(id);
    assert(idx >= 0);
    return createLoad(GlobalParameters, idx,
            name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createGlobalParamStore(
        const std::string& id, llvm::Value* value)
{
    int idx = symbols.getGlobalParameterIndex(id);
    assert(idx >= 0);
    return createStore(GlobalParameters, idx, value, id);
}

llvm::Value* ModelDataIRBuilder::createInitFloatSpeciesAmtGEP(
        const std::string& id, const llvm::Twine& name)
{
    int index = symbols.getFloatingSpeciesInitIndex(id);
    assert(index < symbols.getInitFloatingSpeciesSize());
    assert(index >= 0);
    return createGEP(InitFloatingSpeciesAmounts, index,
            name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createInitBoundarySpeciesAmtGEP(
    const std::string& id, const llvm::Twine& name)
{
    int index = symbols.getBoundarySpeciesInitIndex(id);
    assert(index < symbols.getInitBoundarySpeciesSize());
    assert(index >= 0);
    return createGEP(InitBoundarySpeciesAmounts, index,
        name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createInitFloatSpeciesAmtLoad(
        const std::string& id, const llvm::Twine& name)
{
    Value *gep = createInitFloatSpeciesAmtGEP(id);
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createInitBoundarySpeciesAmtLoad(
    const std::string& id, const llvm::Twine& name)
{
    Value* gep = createInitBoundarySpeciesAmtGEP(id);
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createInitFloatSpeciesAmtStore(
        const std::string& id, llvm::Value* value)
{
    Value *gep = createInitFloatSpeciesAmtGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createInitBoundarySpeciesAmtStore(
    const std::string& id, llvm::Value* value)
{
    Value* gep = createInitBoundarySpeciesAmtGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createInitCompGEP(const std::string& id,
        const llvm::Twine& name)
{
    int index = symbols.getCompartmentInitIndex(id);
    assert(index < symbols.getInitCompartmentSize());
    assert(index >= 0);
    return createGEP(InitCompartmentVolumes, index,
            name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createInitCompLoad(const std::string& id,
        const llvm::Twine& name)
{
    Value *gep = createInitCompGEP(id, name);
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createInitCompStore(const std::string& id,
        llvm::Value* value)
{
    Value *gep = createInitCompGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createInitGlobalParamGEP(const std::string& id,
    const llvm::Twine& name)
{
    int index = symbols.getGlobalParameterInitIndex(id);
    assert(index < symbols.getInitGlobalParameterSize());
    assert(index >= 0);
    return createGEP(InitGlobalParameters, index,
        name.isTriviallyEmpty() ? id : name);
}

llvm::Value* ModelDataIRBuilder::createInitGlobalParamLoad(
        const std::string& id, const llvm::Twine& name)
{
    Value *gep = createInitGlobalParamGEP(id, name);
    return builder.CreateLoad(gep, name);
}

llvm::Value* ModelDataIRBuilder::createInitGlobalParamStore(
        const std::string& id, llvm::Value* value)
{
    Value *gep = createInitGlobalParamGEP(id);
    return builder.CreateStore(value, gep);
}

llvm::Value* ModelDataIRBuilder::createReactionRateLoad(const std::string& id, const llvm::Twine& name)
{
    int idx = symbols.getReactionIndex(id);
    assert(idx >= 0);
    return createLoad(ReactionRates, idx, name);
}

llvm::Value* ModelDataIRBuilder::createReactionRateStore(const std::string& id, llvm::Value* value)
{
    int idx = symbols.getReactionIndex(id);
    assert(idx >= 0);
    return createStore(ReactionRates, idx, value, id);
}

llvm::Value* ModelDataIRBuilder::createStoichiometryStore(uint row, uint col,
        llvm::Value* value, const llvm::Twine& name)
{
    LLVMContext &context = builder.getContext();
    Value *stoichEP = createGEP(Stoichiometry);
    Value *stoich = builder.CreateLoad(stoichEP, "stoichiometry");
    Value *rowVal = ConstantInt::get(Type::getInt32Ty(context), row, true);
    Value *colVal = ConstantInt::get(Type::getInt32Ty(context), col, true);
    return createCSRMatrixSetNZ(builder, stoich, rowVal, colVal, value, name);
}

llvm::Value* ModelDataIRBuilder::createStoichiometryLoad(uint row, uint col,
        const llvm::Twine& name)
{
    LLVMContext &context = builder.getContext();
    Value *stoichEP = createGEP(Stoichiometry);
    Value *stoich = builder.CreateLoad(stoichEP, "stoichiometry");
    Value *rowVal = ConstantInt::get(Type::getInt32Ty(context), row, true);
    Value *colVal = ConstantInt::get(Type::getInt32Ty(context), col, true);
    return createCSRMatrixGetNZ(builder, stoich, rowVal, colVal, name);
}

llvm::Value *ModelDataIRBuilder::createRandomLoad()
{
    Value *randomEP = createGEP(RandomPtr);
    Value *randomPtr = builder.CreateLoad(randomEP, "randomPtr");
    return randomPtr;
}


void ModelDataIRBuilder::validateStruct(llvm::Value* s,
        const char* funcName)
{
    PointerType *pointerType = dyn_cast<PointerType>(s->getType());

    // get the type the pointer refrerences
    Type *type = pointerType ? pointerType->getElementType() : s->getType();

    StructType *structType = dyn_cast<StructType>(type);

    if (structType)
    {
        if (structType->getName().compare(LLVMModelDataName) == 0)
        {
            return;
        }
    }
    std::string str;
    raw_string_ostream err(str);
    err << "error in " << funcName << ", " << "Invalid argument type, expected "
            << LLVMModelDataName << ", but received ";
    type->print(err);
    throw LLVMException(err.str());
}

llvm::StructType *ModelDataIRBuilder::createModelDataStructType(llvm::Module *module,
        llvm::ExecutionEngine *engine, LLVMModelDataSymbols const& symbols)
{
    StructType *structType = getTypeByName(module, LLVMModelDataName);

    if (!structType)
    {
        // these have initial conditions, so need to allocate them twice
        size_t   numIndCompartments = symbols.getIndependentCompartmentSize();
        size_t   numIndFloatingSpecies = symbols.getIndependentFloatingSpeciesSize();
        size_t   numIndBoundarySpecies = symbols.getIndependentBoundarySpeciesSize();
        size_t   numIndGlobalParameters = symbols.getIndependentGlobalParameterSize();

        size_t   numInitCompartments = symbols.getInitCompartmentSize();
        size_t   numInitFloatingSpecies = symbols.getInitFloatingSpeciesSize();
        size_t   numInitBoundarySpecies = symbols.getInitBoundarySpeciesSize();
        size_t   numInitGlobalParameters = symbols.getInitGlobalParameterSize();

        // no initial conditions for these
        size_t   numRateRules = symbols.getRateRuleSize();
        size_t   numReactions = symbols.getReactionSize();

        LLVMContext &context = module->getContext();

        Type *csrSparseType = getCSRSparseStructType(module, engine);
        Type *csrSparsePtrType = csrSparseType->getPointerTo();

        Type *int32Type = Type::getInt32Ty(context);
        Type *doubleType = Type::getDoubleTy(context);
        Type *doublePtrType = Type::getDoublePtrTy(context);

        // LLVM does not appear to have a true void ptr, so just use a pointer
        // to a byte, pointers are all the same size anyway.
        // used for the LLVMModelData::random which is not accessed by
        // generated llvm code anyway.
        Type *voidPtrType = Type::getInt8PtrTy(context);

        std::vector<Type*> elements;

        elements.push_back(int32Type);        // 0      unsigned                 size;
        elements.push_back(int32Type);        // 1      unsigned                 flags;
        elements.push_back(doubleType);       // 2      double                   time;
        elements.push_back(int32Type);        // 3      int                      numIndCompartments;
        elements.push_back(int32Type);        // 4      int                      numIndFloatingSpecies;
        elements.push_back(int32Type);        // 5      int                      numIndBoundarySpecies;
        elements.push_back(int32Type);        // 6      int                      numIndGlobalParameters;
        elements.push_back(int32Type);        // 7      int                      numRateRules;
        elements.push_back(int32Type);        // 8      int                      numReactions;

        elements.push_back(int32Type);        // 9      int                      numInitCompartments;
        elements.push_back(int32Type);        // 10     int                      numInitFloatingSpecies;
        elements.push_back(int32Type);        // 11     int                      numInitBoundarySpecies;
        elements.push_back(int32Type);        // 12     int                      numInitGlobalParameters;

        elements.push_back(csrSparsePtrType); // 13     dcsr_matrix              stoichiometry;
        elements.push_back(voidPtrType);      // 14     void*                    random;
        elements.push_back(int32Type);        // 15     int                      numEvents;
        elements.push_back(int32Type);        // 16     int                      numPiecewiseTriggers;
        elements.push_back(int32Type);        // 17     int                      stateVectorSize;
        elements.push_back(doublePtrType);    // 18     double*                  stateVector;
        elements.push_back(doublePtrType);    // 19     double*                  stateVectorRate;
        elements.push_back(doublePtrType);    // 20     double*                  rateRuleRates;
        elements.push_back(doublePtrType);    // 21     double*                  floatingSpeciesAmountRates;

        elements.push_back(doublePtrType);    // 22     double*                  compartmentVolumesAlias;
        elements.push_back(doublePtrType);    // 23     double*                  compartmentVolumesInitAlias;
        elements.push_back(doublePtrType);    // 24     double*                  floatingSpeciesAmountsInitAlias
        elements.push_back(doublePtrType);    // 25     double*                  boundarySpeciesAmountsAlias;
        elements.push_back(doublePtrType);    // 26     double*                  boundarySpeciesAmountsInitAlias;
        elements.push_back(doublePtrType);    // 27     double*                  globalParametersAlias
        elements.push_back(doublePtrType);    // 28     double*                  globalParametersInitAlias
        elements.push_back(doublePtrType);    // 29     double*                  reactionRatesAlias

        elements.push_back(doublePtrType);    // 30     double*                  rateRuleValuesAlias
        elements.push_back(doublePtrType);    // 31     double*                  floatingSpeciesAmountsAlias

        elements.push_back(ArrayType::get(doubleType, numIndCompartments));     // 32 CompartmentVolumes
        elements.push_back(ArrayType::get(doubleType, numInitCompartments));    // 33 initCompartmentVolumes
        elements.push_back(ArrayType::get(doubleType, numInitFloatingSpecies)); // 34 initFloatingSpeciesAmounts
        elements.push_back(ArrayType::get(doubleType, numIndBoundarySpecies));  // 35 boundarySpeciesAmounts
        elements.push_back(ArrayType::get(doubleType, numInitBoundarySpecies)); // 36 initBoundarySpeciesAmounts
        elements.push_back(ArrayType::get(doubleType, numIndGlobalParameters)); // 37 globalParameters
        elements.push_back(ArrayType::get(doubleType, numInitGlobalParameters));// 38 initGlobalParameters
        elements.push_back(ArrayType::get(doubleType, numReactions));           // 39 reactionRates
        elements.push_back(ArrayType::get(doubleType, numRateRules));           // 40 rateRuleValues
        elements.push_back(ArrayType::get(doubleType, numIndFloatingSpecies));  // 41 floatingSpeciesAmounts

        // creates a named struct,
        // the act of creating a named struct should
        // add the struct to the module definttion, at least
        // the code in llvm 3.3 does this.
        structType = StructType::create(context, elements, LLVMModelDataName);

        // make sure we can get the struct in the future
        assert(getTypeByName(module, LLVMModelDataName) &&
                "Could not get LLVMModelData struct from llvm module after creating it");
    }
    return structType;
}

llvm::StructType *ModelDataIRBuilder::getStructType(llvm::Module *module)
{
    StructType *structType = getTypeByName(module, LLVMModelDataName);

    if (structType == 0)
    {
        throw_llvm_exception("Could not get LLVMModelData struct type from llvm Module, "
                "createModelDataStructType probably has not been called.");
    }

    return structType;
}

unsigned ModelDataIRBuilder::getModelDataSize(llvm::Module *module, const DataLayout& dl)
{
    StructType *structType = getStructType(module);
	uint64_t llvm_size = dl.getTypeStoreSize(structType);

    // the model data struct will NEVER be bigger than a 32 bit pointer!
    return (unsigned)llvm_size;

    /*
     printf("TestStruct size: %i, , LLVM Size: %i\n", sizeof(ModelData), llvm_size);

     printf("C++ bool size: %i, LLVM bool size: %i\n", sizeof(bool), dl->getTypeStoreSize(boolType));
     printf("C++ bool* size: %i, LLVM bool* size: %i\n", sizeof(bool*), dl->getTypeStoreSize(boolPtrType));
     printf("C++ char** size: %i, LLVM char** size: %i\n", sizeof(char**), dl->getTypeStoreSize(charStarStarType));
     printf("C++ void* size: %i, LLVM void* size: %i\n", sizeof(void*), dl->getTypeStoreSize(voidPtrType));

     printf("C++ EventDelayHandler* size: %i\n", sizeof(EventDelayHandler*));
     printf("C++ EventAssignmentHandler* size: %i\n", sizeof(EventAssignmentHandler* ));
     printf("C++ ComputeEventAssignmentHandler* size: %i\n", sizeof(ComputeEventAssignmentHandler*));
     printf("C++ PerformEventAssignmentHandler* size: %i\n", sizeof(PerformEventAssignmentHandler*));

     */

}


/*****************************************************************************/
/*****************************************************************************/
/*****************              TESTING STUFF              *******************/
/*****************************************************************************/
/*****************************************************************************/



LLVMModelDataIRBuilderTesting::LLVMModelDataIRBuilderTesting(LLVMModelDataSymbols const& symbols,
        llvm::IRBuilder<> &b) :
        symbols(symbols),
        builder(b)
{
}


void LLVMModelDataIRBuilderTesting::createAccessors(Module *module)
{
    const std::string getSizeName = "get_size";
    Function* getSizeFunc = module->getFunction(getSizeName);
    if(!getSizeFunc)
    {
        LLVMContext &context = module->getContext();
        StructType *structType = ModelDataIRBuilder::getStructType(module);
        std::vector<Type*> getArgTypes(1, PointerType::get(structType, 0));
        FunctionType *getFuncType = FunctionType::get(Type::getInt32Ty(context),
                getArgTypes, false);
        getSizeFunc = Function::Create(getFuncType, Function::ExternalLinkage,
                getSizeName, module);

        BasicBlock *getBlock = BasicBlock::Create(context, "entry", getSizeFunc);
        builder.SetInsertPoint(getBlock);
        std::vector<Value*> getArgValues;
        for(Function::arg_iterator i = getSizeFunc->arg_begin();
                i != getSizeFunc->arg_end(); i++)
        {
            getArgValues.push_back(i);
        }

        ModelDataIRBuilder mdbuilder (getArgValues[0], symbols, builder);



        Value *gep = mdbuilder.createGEP(Size);

        Value *getRet = builder.CreateLoad(gep);

        builder.CreateRet(getRet);

        verifyFunction(*getSizeFunc);

        //getSizeFunc->dump();

    }

}

llvm::CallInst* LLVMModelDataIRBuilderTesting::createDispChar(llvm::Value* intVal)
{
    return builder.CreateCall(getDispCharDecl(ModelDataIRBuilder::getModule(builder, __FUNC__)), intVal);
}

llvm::CallInst* LLVMModelDataIRBuilderTesting::createDispDouble(llvm::Value* doubleVal)
{
    return builder.CreateCall(getDispDoubleDecl(ModelDataIRBuilder::getModule(builder, __FUNC__)), doubleVal);
}

llvm::CallInst* LLVMModelDataIRBuilderTesting::createDispInt(llvm::Value* intVal)
{
    return builder.CreateCall(getDispIntDecl(ModelDataIRBuilder::getModule(builder, __FUNC__)), intVal);
}


std::pair<Function*, Function*> LLVMModelDataIRBuilderTesting::createFloatingSpeciesAccessors(
        llvm::Module* module, const std::string id)
{
    const std::string getName = "get_floatingspecies_conc_" + id;
    const std::string setName = "set_floatingspecies_conc_" + id;

    std::pair<Function*, Function*> result(module->getFunction(getName),
            module->getFunction(setName));

    if (!result.first || !result.second)
    {
        LLVMContext &context = module->getContext();
        StructType *structType = ModelDataIRBuilder::getStructType(module);
        std::vector<Type*> getArgTypes(1, PointerType::get(structType, 0));
        FunctionType *getFuncType = FunctionType::get(
                Type::getDoubleTy(context), getArgTypes, false);
        result.first = Function::Create(getFuncType, Function::ExternalLinkage,
                getName, module);

        BasicBlock *block = BasicBlock::Create(context, "entry",
                result.first);
        builder.SetInsertPoint(block);
        std::vector<Value*> getArgValues;
        for (Function::arg_iterator i = result.first->arg_begin();
                i != result.first->arg_end(); i++)
        {
            getArgValues.push_back(i);

        }

        ModelDataIRBuilder mdbuilder (getArgValues[0], symbols, builder);


        //Value *getEP = mdbuilder.createFloatSpeciesConcGEP(id);

        //Value *getRet = builder.CreateLoad(getEP);

        //builder.CreateRet(getRet);

        verifyFunction(*result.first);

        //result.first->dump();

        std::vector<Type*> setArgTypes;
        setArgTypes.push_back(PointerType::get(structType, 0));
        setArgTypes.push_back(Type::getDoubleTy(context));
        FunctionType *setFuncType = FunctionType::get(Type::getVoidTy(context),
                setArgTypes, false);
        result.second = Function::Create(setFuncType, Function::ExternalLinkage,
                setName, module);

        block = BasicBlock::Create(context, "entry",
                result.second);
        builder.SetInsertPoint(block);
        std::vector<Value*> setArgValues;
        for (Function::arg_iterator i = result.second->arg_begin();
                i != result.second->arg_end(); i++)
        {
            setArgValues.push_back(i);

        }


        //Value *val = mdbuilder.createFloatSpeciesConcStore(id, setArgValues[1]);

        builder.CreateRetVoid();

        verifyFunction(*result.second);

        //result.second->dump();

        std::cout << "pause...\n";

    }

    return result;
}

llvm::Function* LLVMModelDataIRBuilderTesting::getDispCharDecl(llvm::Module* module)
{
    Function *f = module->getFunction("dispChar");
    if (f == 0) {
        std::vector<Type*> args(1, Type::getInt8Ty(module->getContext()));
        FunctionType *funcType = FunctionType::get(Type::getVoidTy(module->getContext()), args, false);
        f = Function::Create(funcType, Function::ExternalLinkage, "dispChar", module);
    }
    return f;
}



llvm::Function* LLVMModelDataIRBuilderTesting::getDispDoubleDecl(llvm::Module* module)
{
    Function *f = module->getFunction("dispDouble");
    if (f == 0) {
        std::vector<Type*> args(1, Type::getDoubleTy(module->getContext()));
        FunctionType *funcType = FunctionType::get(Type::getVoidTy(module->getContext()), args, false);
        f = Function::Create(funcType, Function::ExternalLinkage, "dispDouble", module);
    }
    return f;
}


llvm::Function* LLVMModelDataIRBuilderTesting::getDispIntDecl(llvm::Module* module)
{
    Function *f = module->getFunction("dispInt");
    if (f == 0) {
        std::vector<Type*> args(1, Type::getInt32Ty(module->getContext()));
        FunctionType *funcType = FunctionType::get(Type::getVoidTy(module->getContext()), args, false);
        f = Function::Create(funcType, Function::ExternalLinkage, "dispInt", module);
    }
    return f;
}


} /* namespace rr */



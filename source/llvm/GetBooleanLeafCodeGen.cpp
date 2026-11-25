/*
 * GetBooleanLeafCodeGen.cpp
 */
#pragma hdrstop
#include "GetBooleanLeafCodeGen.h"
#include "LLVMException.h"
#include "ModelDataSymbolResolver.h"
#include "KineticLawParameterResolver.h"
#include "rrLogger.h"

#include <Poco/Logger.h>
#include <vector>

using namespace llvm;

using namespace libsbml;

namespace rrllvm
{
    const char* GetBooleanLeafCodeGen::FunctionName = "getBooleanLeaf";
    const char* GetBooleanLeafCodeGen::IndexArgName = "triggerIndx";

    GetBooleanLeafCodeGen::GetBooleanLeafCodeGen(const ModelGeneratorContext& mgc)
        : CodeGenBase<GetBooleanLeafCodeGen_FunctionPtr>(mgc)
        , booleanLeaves(mgc.getBooleanLeaves())
    {
    };

    llvm::Value* GetBooleanLeafCodeGen::codeGen()
    {
        llvm::Type* argTypes[] = {
            llvm::PointerType::getUnqual(this->context),
            llvm::Type::getInt32Ty(this->context)
        };

        const char* argNames[] = {
            "modelData", IndexArgName
        };

        llvm::Value* args[] = { 0, 0 };

        llvm::Type* retType = getRetType();

        llvm::BasicBlock* entry = this->codeGenHeader(FunctionName, retType,
            argTypes, argNames, args);

        ModelDataLoadSymbolResolver resolver(args[0], this->modelGenContext);

        llvm::BasicBlock* def = llvm::BasicBlock::Create(this->context, "default", this->function);
        this->builder.SetInsertPoint(def);

        llvm::Value* defRet = createRet(0);
        this->builder.CreateRet(defRet);

        // the switch is the terminator of the entry block
        this->builder.SetInsertPoint(entry);

        llvm::SwitchInst* s = this->builder.CreateSwitch(args[1], def, booleanLeaves->size());

        for (uint i = 0; i < booleanLeaves->size(); ++i)
        {
            char block_name[64];
            std::snprintf(block_name, 64, "booleanLeaf_%i_block", i);
            llvm::BasicBlock* block = llvm::BasicBlock::Create(this->context, block_name, this->function);
            this->builder.SetInsertPoint(block);
            resolver.flushCache();

            // A leaf from a kinetic law may use the local parameters of that law.
            LoadSymbolResolver* leafResolver = &resolver;
            std::unique_ptr<KineticLawParameterResolver> kineticLawResolver;
            const libsbml::KineticLaw* kineticLaw = this->modelGenContext.getBooleanLeafKineticLaw(i);
            if (kineticLaw != NULL) {
                kineticLawResolver.reset(new KineticLawParameterResolver(resolver, *kineticLaw, this->builder));
                leafResolver = kineticLawResolver.get();
            }
            ASTNodeCodeGen astCodeGen(this->builder, *leafResolver, this->modelGenContext, args[0]);

            llvm::Value* value = astCodeGen.codeGenBoolean((*booleanLeaves)[i]);

            value = createRet(value);

            this->builder.CreateRet(value);
            s->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(this->context), i), block);
        }

        return this->verifyFunction();
    }

    llvm::Type* GetBooleanLeafCodeGen::getRetType()
    {
        return llvm::Type::getInt8Ty(context);
    };

    llvm::Value* GetBooleanLeafCodeGen::createRet(llvm::Value* value)
    {
        if (!value)
        {
            return llvm::ConstantInt::get(getRetType(), 0xff, false);
        }
        else
        {
            return builder.CreateIntCast(value, getRetType(), false);
        }
    }

} /* namespace rrllvm */

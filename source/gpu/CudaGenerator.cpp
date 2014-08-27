// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * CudaGenerator.cpp
 *
 *  Created on: Aug 21, 2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "CudaGenerator.hpp"

#include "Poco/SharedLibrary.h"

#include <fstream>
#include <sstream>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

void CudaGenerator::generate(GPUSimExecutableModel& model, double h) {
    CudaModule mod;
    std::string entryName = "cuEntryPoint";

    // get the size of the state vector
    int n = model.getStateVector(NULL);

    // construct the DOM

    // macros
    Macro* N = mod.addMacro(Macro("N", std::to_string(n)));

    Macro* RK4ORDER = mod.addMacro(Macro("RK4ORDER", "4"));

    Macro* RK_COEF_LEN = mod.addMacro(Macro("RK_COEF_LEN", "RK4ORDER*RK4ORDER*N"));
    Macro* RK_COEF_OFFSET = mod.addMacro(Macro("RK_COEF_OFFSET", "gen*RK4ORDER*N + idx*N + component", "gen", "idx", "component"));
//     Macro* RK_COEF_SIZE = mod.addMacro(Macro("RK_COEF_SIZE", ""));

    mod.addMacro(Macro("RK_STATE_VEC_LEN", "RK4ORDER*N"));
    mod.addMacro(Macro("RK_STATE_VEC_OFFSET", "idx*N + component", "idx", "component"));

    mod.addMacro(Macro("RK_TIME_VEC_LEN", "RK4ORDER"));

    // typedef for float or double
    Type* RKReal = TypedefStatement::downcast(mod.addStatement(StatementPtr(new TypedefStatement(BaseTypes::getTp(BaseTypes::FLOAT), "RKReal"))))->getAlias();

    CudaKernel* kernel = CudaKernel::downcast(mod.addFunction(CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(RKReal, "h")})));

    // generate the kernel
    {
        // declare shared memory
//         kernel->addStatement(ExpressionPtr(new CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "k")), true)));
        Variable* k = CudaVariableDeclarationExpression::downcast(ExpressionStatement::insert(*kernel, CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "k")), true))->getExpression())->getVariable();

        Variable* f = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(BaseTypes::get().addPointer(RKReal), "f")), ReferenceExpression(ArrayIndexExpression(k, MacroExpression(RK_COEF_LEN)))))->getExpression())->getVariable();

        // printf
        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));

        // init k
        ForStatement* init_k_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

        Variable* j = init_k_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

        init_k_loop->setInitExp(ExpressionPtr(new VariableInitExpression(j, ExpressionPtr(new LiteralIntExpression(0)))));
        init_k_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(j)), ExpressionPtr(new MacroExpression(RK4ORDER)))));
        init_k_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(j)))));

//         Variable* z = init_k_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "z"));

//         ExpressionStatement::insert(init_k_loop->getBody(), AssignmentExpression(VariableRefExpression(z), LiteralIntExpression(1)));
//         init_k_loop->getBody().addStatement(StatementPtr(new ExpressionStatement(AssignmentExpression(VariableRefExpression(z), LiteralIntExpression(1)))));

        AssignmentExpression* k_init_assn =
            AssignmentExpression::downcast(ExpressionStatement::insert(init_k_loop->getBody(), AssignmentExpression(
            ArrayIndexExpression(k,
            MacroExpression(RK_COEF_OFFSET,
            VariableRefExpression(j),
            kernel->getThreadIdx(CudaKernel::IndexComponent::x),
            kernel->getBlockIdx (CudaKernel::IndexComponent::x))),
            LiteralIntExpression(0)))->getExpression());

        ExpressionStatement::insert(init_k_loop->getBody(), FunctionCallExpression(
            mod.getPrintf(),
            StringLiteralExpression("k[RK_COEF_OFFSET(%d, %d, %d)] = %f\\n"),
            VariableRefExpression(j),
            kernel->getThreadIdx(CudaKernel::IndexComponent::x),
            kernel->getBlockIdx (CudaKernel::IndexComponent::x),
            k_init_assn->getLHS()->clone()
          ));
    }

    CudaFunction* entry = mod.addFunction(CudaFunction(entryName, BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(RKReal, "h")}));
//     entry->setHasCLinkage(true);

    // generate the entry
    {
        // printf
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in cuda\\n")))));

        // call the kernel
        CudaKernelCallExpression* kernel_call = CudaKernelCallExpression::downcast(ExpressionStatement::insert(entry, CudaKernelCallExpression(n, 4, 1, kernel, VariableRefExpression(entry->getPositionalParam(0))))->getExpression());

        kernel_call->setSharedMemSize(ProductExpression(MacroExpression(RK_COEF_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))));

        // call cudaDeviceSynchronize
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaDeviceSynchronize())));
    }

    // serialize the module to a document

    {
        Serializer s("/tmp/rr_cuda_model.cu");
        mod.serialize(s);
    }

    // compile the module

    FILE* pp = popen("nvcc -D__CUDACC__ -ccbin gcc -m32 -I/home/jkm/devel/src/roadrunner/source --ptxas-options=-v --compiler-options '-fPIC' -Drr_cuda_model_EXPORTS -Xcompiler ,\"-fPIC\",\"-fPIC\",\"-g\" -DNVCC --shared -o /tmp/rr_cuda_model.so /tmp/rr_cuda_model.cu 2>&1 >/dev/null", "r");

#define SBUFLEN 512
    char sbuf[SBUFLEN];
    fgets(sbuf, SBUFLEN, pp);

    int code = pclose(pp);
    pp = NULL;
    std::cerr << "Return code: " << code << "\n";

    if(code/256) {
        std::stringstream ss;
        ss << "Compiler errors:\n" << sbuf << "\n";
        throw_gpusim_exception(ss.str());
    }

    // get the mangled name of the entry point

    std::string demangleCmd = "nm -g /tmp/rr_cuda_model.so | grep -ohe '_.*" + entryName + "[^\\s]*$'";
    pp = popen(demangleCmd.c_str(), "r");

    fgets(sbuf, SBUFLEN, pp);
    std::string entryMangled{sbuf};

    code = pclose(pp);

    if(code/256) {
        std::stringstream ss;
        ss << "Could not find symbol: " << entryName << "\n";
        throw_gpusim_exception(ss.str());
    }

    // strip newline

    while(entryMangled.back() == '\n' && entryMangled.size())
        entryMangled = std::string(entryMangled,0,entryMangled.size()-1);

    // load the module

    std::string libname = "/tmp/rr_cuda_model.so";
    Poco::SharedLibrary so;
    try {
        so.load(libname);
    } catch(Poco::LibraryLoadException e) {
        throw_gpusim_exception("Cannot load lib: " + e.message());
    }

    Log(Logger::LOG_DEBUG) << "Loading symbol " << entryMangled << " in " << libname;
    if(!so.hasSymbol(entryMangled))
        throw_gpusim_exception("Lib " + libname + " has no symbol \"" + entryMangled + "\"");

    Log(Logger::LOG_TRACE) << "Entering CUDA code";
    typedef void (*EntryPointSig)(float);
    // ensure that the function has the correct signature
    assert(sizeof(float) == RKReal->Sizeof());
    EntryPointSig entryPoint;
    entryPoint = (EntryPointSig)so.getSymbol(entryMangled);

    return;
    entryPoint(h);

}

} // namespace dom

} // namespace rrgpu

} // namespace rr

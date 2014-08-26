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

void CudaGenerator::generate(const GPUSimModel& model) {
    CudaModule mod;
    std::string entryName = "cuEntryPoint";

    // construct the DOM

    // macros
    Macro* RK4ORDER = mod.addMacro(Macro("RK4ORDER", "4"));

    mod.addMacro(Macro("RK_COEF_LEN", "RK4ORDER*RK4ORDER*n"));
    mod.addMacro(Macro("RK_COEF_OFFSET", "gen*RK4BLOCKS*n + idx*n + component", "gen", "idx", "component"));

    mod.addMacro(Macro("RK_STATE_VEC_LEN", "RK4ORDER*n"));
    mod.addMacro(Macro("RK_STATE_VEC_OFFSET", "idx*n + component", "idx", "component"));

    mod.addMacro(Macro("RK_TIME_VEC_LEN", "RK4ORDER"));

    // typedef for float
    Type* RKReal = TypedefStatement::downcast(mod.addStatement(StatementPtr(new TypedefStatement(BaseTypes::getTp(BaseTypes::FLOAT), "RKReal"))))->getAlias();

    CudaKernelPtr kernel(new CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID)));

    // generate the kernel
    {
        // declare shared memory
        kernel->addStatement(ExpressionPtr(new CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "k")), true)));

        // printf
        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));

        // init k
        ForStatement* init_k_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

        Variable* j = init_k_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

        init_k_loop->setInitExp(ExpressionPtr(new VariableInitExpression(j, ExpressionPtr(new LiteralIntExpression(1)))));
        init_k_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(j)), ExpressionPtr(new MacroExpression(RK4ORDER)))));
        init_k_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(j)))));

//         init_k_loop->getBody()->addStatement(StatementPtr())
    }

    CudaModule::CudaFunctionPtr entry(new CudaFunction(entryName, BaseTypes::getTp(BaseTypes::VOID)));
//     entry->setHasCLinkage(true);

    // printf
    entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in cuda\\n")))));

    // call the kernel
    ExpressionPtr calltokern(new CudaKernelCallExpression(1, 10, 1, kernel.get()));
    entry->addStatement(StatementPtr(new ExpressionStatement(std::move(calltokern))));

    // call cudaDeviceSynchronize
    entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaDeviceSynchronize())));

    mod.addFunction(std::move(kernel));
    mod.addFunction(std::move(entry));

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
    typedef void (*EntryPointSig)();
    EntryPointSig entryPoint;
    entryPoint = (EntryPointSig)so.getSymbol(entryMangled);

    entryPoint();

}

} // namespace dom

} // namespace rrgpu

} // namespace rr

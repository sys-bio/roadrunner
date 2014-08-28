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

class CudaGeneratorImpl {
public:
    typedef CudaGenerator::EntryPointSig EntryPointSig;

    void generate(GPUSimExecutableModel& model);

    EntryPointSig getEntryPoint();

protected:
    EntryPointSig entry_ = nullptr;
    Poco::SharedLibrary so_;
};

CudaGenerator::CudaGenerator()
  :  impl_(new CudaGeneratorImpl()) {

}

CudaGenerator::~CudaGenerator() {

}

void CudaGenerator::generate(GPUSimExecutableModel& model) {
    impl_->generate(model);
}

CudaGenerator::EntryPointSig CudaGenerator::getEntryPoint() {
    return impl_->getEntryPoint();
}

CudaGeneratorImpl::EntryPointSig CudaGeneratorImpl::getEntryPoint() {
    if (!entry_)
        throw_gpusim_exception("No entry point set (did you forget to call generate first?)");
    return entry_;
}

void CudaGeneratorImpl::generate(GPUSimExecutableModel& model) {
    CudaModule mod;
    std::string entryName = "cuEntryPoint";

    // get the size of the state vector
    int n = model.getStateVector(NULL);

    // construct the DOM

    // macros
    Macro* N = mod.addMacro(Macro("N", std::to_string(n)));

    Macro* RK4ORDER = mod.addMacro(Macro("RK4ORDER", "4"));

    Macro* RK_COEF_LEN = mod.addMacro(Macro("RK_COEF_LEN", "RK4ORDER*RK4ORDER*N"));
    Macro* RK_COEF_GET_OFFSET = mod.addMacro(Macro("RK_COEF_GET_OFFSET", "gen*RK4ORDER*N + idx*N + component", "gen", "idx", "component"));
//     Macro* RK_COEF_SIZE = mod.addMacro(Macro("RK_COEF_SIZE", ""));

    Macro* RK_GET_INDEX = mod.addMacro(Macro("RK_GET_INDEX", "(threadIdx.x%N)"));

    Macro* RK_GET_COMPONENT = mod.addMacro(Macro("RK_GET_COMPONENT", "(threadIdx.x/32)"));

    Macro* RK_STATE_VEC_LEN = mod.addMacro(Macro("RK_STATE_VEC_LEN", "RK4ORDER*N"));
    Macro* RK_STATE_VEC_GET_OFFSET = mod.addMacro(Macro("RK_STATE_VEC_GET_OFFSET", "idx*N + component", "idx", "component"));

    Macro* RK_TIME_VEC_LEN = mod.addMacro(Macro("RK_TIME_VEC_LEN", "RK4ORDER"));

    // typedef for float or double
    Type* RKReal = TypedefStatement::downcast(mod.addStatement(StatementPtr(new TypedefStatement(BaseTypes::getTp(BaseTypes::FLOAT), "RKReal"))))->getAlias();

    Type* pRKReal = BaseTypes::get().addPointer(RKReal);

    Type* pRKReal_volatile = BaseTypes::get().addVolatile(pRKReal);

    CudaFunction* PrintCoefs = mod.addFunction(CudaFunction("PrintCoefs", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(pRKReal, "k")}));

    PrintCoefs->setIsDeviceFun(true);

    // printf
    PrintCoefs->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("PrintCoefs\\n")))));

    CudaKernel* kernel = CudaKernel::downcast(mod.addFunction(CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(RKReal, "h"), FunctionParameter(pRKReal_volatile, "k_global")})));

    // the step size
    const FunctionParameter* h = kernel->getPositionalParam(0);
    assert(h->getType()->isIdentical(RKReal) && "h has wrong type - signature of kernel changed? ");

    // generate the kernel
    {
        // declare shared memory
//         kernel->addStatement(ExpressionPtr(new CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "k")), true)));
        Variable* shared_buf = CudaVariableDeclarationExpression::downcast(ExpressionStatement::insert(*kernel, CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "shared_buf")), true))->getExpression())->getVariable();

        Variable* k = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "k")), ReferenceExpression(ArrayIndexExpression(shared_buf, LiteralIntExpression(0)))))->getExpression())->getVariable();

        Variable* f = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "f")), ReferenceExpression(ArrayIndexExpression(k, MacroExpression(RK_COEF_LEN)))))->getExpression())->getVariable();

        Variable* t = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "t")), ReferenceExpression(ArrayIndexExpression(f, MacroExpression(RK_STATE_VEC_LEN)))))->getExpression())->getVariable();

        // printf
        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));

        {
            // loop for initializing k (the RK coefficients)
            ForStatement* init_k_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

            Variable* j = init_k_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

            init_k_loop->setInitExp(ExpressionPtr(new VariableInitExpression(j, ExpressionPtr(new LiteralIntExpression(0)))));
            init_k_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(j)), ExpressionPtr(new MacroExpression(RK4ORDER)))));
            init_k_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(j)))));

            // init expression for k
            AssignmentExpression* k_init_assn =
                AssignmentExpression::downcast(ExpressionStatement::insert(init_k_loop->getBody(), AssignmentExpression(
                ArrayIndexExpression(k,
                MacroExpression(RK_COEF_GET_OFFSET,
                VariableRefExpression(j),
                MacroExpression(RK_GET_INDEX),
                MacroExpression(RK_GET_COMPONENT))),
                LiteralIntExpression(0)))->getExpression());

            // printf showing the init'd value of k
            ExpressionStatement::insert(init_k_loop->getBody(), FunctionCallExpression(
                mod.getPrintf(),
                StringLiteralExpression("k[RK_COEF_GET_OFFSET(%d, %d, %d)] = %f\\n"),
                VariableRefExpression(j),
                MacroExpression(RK_GET_INDEX),
                MacroExpression(RK_GET_COMPONENT),
                k_init_assn->getLHS()->clone()
            ));
        }

        // initialize f (the state vector)
        AssignmentExpression* f_init_assn =
            AssignmentExpression::downcast(ExpressionStatement::insert(*kernel,
            AssignmentExpression(
                ArrayIndexExpression(f,
        MacroExpression(RK_STATE_VEC_GET_OFFSET,
        MacroExpression(RK_GET_INDEX),
        MacroExpression(RK_GET_COMPONENT))),
              LiteralIntExpression(0)))->getExpression());

        ExpressionStatement::insert(*kernel, FunctionCallExpression(
            mod.getPrintf(),
            StringLiteralExpression("f[RK_STATE_VEC_GET_OFFSET(%d, %d)] = %f\\n"),
            kernel->getThreadIdx(CudaKernel::IndexComponent::x),
            kernel->getBlockIdx (CudaKernel::IndexComponent::x),
            f_init_assn->getLHS()->clone()
          ));

        // initialize t (the time vector)
        AssignmentExpression* t_init_assn0 =
            AssignmentExpression::downcast(ExpressionStatement::insert(*kernel,
            AssignmentExpression(
                ArrayIndexExpression(t, LiteralIntExpression(0)),
              LiteralIntExpression(0)))->getExpression());

        AssignmentExpression* t_init_assn1 =
            AssignmentExpression::downcast(ExpressionStatement::insert(*kernel,
            AssignmentExpression(
                ArrayIndexExpression(t, LiteralIntExpression(1)),
              ProductExpression(RealLiteralExpression(0.5), VariableRefExpression(h))))->getExpression());

        AssignmentExpression* t_init_assn2 =
            AssignmentExpression::downcast(ExpressionStatement::insert(*kernel,
            AssignmentExpression(
                ArrayIndexExpression(t, LiteralIntExpression(2)),
              ProductExpression(RealLiteralExpression(0.5), VariableRefExpression(h))))->getExpression());

        AssignmentExpression* t_init_assn3 =
            AssignmentExpression::downcast(ExpressionStatement::insert(*kernel,
            AssignmentExpression(
                ArrayIndexExpression(t, LiteralIntExpression(3)),
              VariableRefExpression(h)))->getExpression());

        // print coefs
        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(PrintCoefs, VariableRefExpression(k))));

        // loop
        {
            ForStatement* update_coef_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

            Variable* j = update_coef_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

            update_coef_loop->setInitExp(ExpressionPtr(new VariableInitExpression(j, ExpressionPtr(new LiteralIntExpression(0)))));
            update_coef_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(j)), ExpressionPtr(new MacroExpression(RK4ORDER)))));
            update_coef_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(j)))));

            SwitchStatement* component_switch = SwitchStatement::insert(update_coef_loop->getBody(), MacroExpression(RK_GET_COMPONENT));

            component_switch->addCase(LiteralIntExpression(0));

            AssignmentExpression* k_assn =
                AssignmentExpression::downcast(ExpressionStatement::insert(component_switch->getBody(), AssignmentExpression(
                ArrayIndexExpression(k,
                MacroExpression(RK_COEF_GET_OFFSET,
                VariableRefExpression(j),
                MacroExpression(RK_GET_INDEX),
                MacroExpression(RK_GET_COMPONENT))),
                LiteralIntExpression(0)))->getExpression());
        }
    }

    CudaFunction* entry = mod.addFunction(CudaFunction(entryName, BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(RKReal, "h")}));
//     entry->setHasCLinkage(true);

    // generate the entry
    {
        // printf
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in cuda\\n")))));

        // allocate mem for coefs
        Variable* k_global =
            VariableDeclarationExpression::downcast(ExpressionStatement::insert(entry,
            VariableDeclarationExpression(entry->addVariable(Variable(pRKReal, "k_global"))))->getExpression())->getVariable();
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaMalloc(),
        ReferenceExpression(VariableRefExpression(k_global)),
        ProductExpression(MacroExpression(RK_COEF_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))))));

        // call the kernel
        CudaKernelCallExpression* kernel_call = CudaKernelCallExpression::downcast(ExpressionStatement::insert(entry, CudaKernelCallExpression(1, 1, 1, kernel, VariableRefExpression(entry->getPositionalParam(0)), VariableRefExpression(k_global)))->getExpression());

        kernel_call->setNumThreads(ProductExpression(MacroExpression(N), LiteralIntExpression(32)));

        kernel_call->setSharedMemSize(SumExpression(SumExpression(
            ProductExpression(MacroExpression(RK_COEF_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))), // Coefficient size
            ProductExpression(MacroExpression(RK_STATE_VEC_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal)))), // State vector size
            ProductExpression(MacroExpression(RK_TIME_VEC_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))) // Time vector size
            ));

        // free the global coef mem
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaFree(), VariableRefExpression(k_global))));

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
        ss << "nvcc code: " << code << "\n";
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

    try {
        so_.load(libname);
    } catch(Poco::LibraryLoadException e) {
        throw_gpusim_exception("Cannot load lib: " + e.message());
    }

    Log(Logger::LOG_DEBUG) << "Loading symbol " << entryMangled << " in " << libname;
    if(!so_.hasSymbol(entryMangled))
        throw_gpusim_exception("Lib " + libname + " has no symbol \"" + entryMangled + "\"");

    Log(Logger::LOG_TRACE) << "Entering CUDA code";

    // ensure that the function has the correct signature
    assert(sizeof(float) == RKReal->Sizeof());

    entry_ = (EntryPointSig)so_.getSymbol(entryMangled);

}

} // namespace dom

} // namespace rrgpu

} // namespace rr

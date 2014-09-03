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

class CudaGeneratorSBML {
public:
    /**
     * @brief
     * @details Given a component of the state vector,
     * determine which side (if any) of the reaction
     * it is on
     */
    int getReactionSide(const std::string& species_id);
};

class CudaGeneratorImpl {
public:
    typedef CudaGenerator::EntryPointSig EntryPointSig;

    /// Ctor
    CudaGeneratorImpl(GPUSimExecutableModel& mod)
      : mod_(mod) {}

    ExpressionPtr generateReactionRateExp(const Reaction* r);

    ExpressionPtr accumulate(ExpressionPtr&& sum, ExpressionPtr&& item, bool invert);

    ExpressionPtr generateEvalExp(int component, int rk_index);

    ExpressionPtr getVecRK(const FloatingSpecies* s);

# if GPUSIM_MODEL_USE_SBML
    ExpressionPtr generateExpForSBMLASTNode(const Reaction* r, const libsbml::ASTNode* node);
# else
# error "Need SBML"
# endif

    void generate();

    EntryPointSig getEntryPoint();

protected:

    ExpressionPtr getK(ExpressionPtr&& index, ExpressionPtr&& component) {
        return ExpressionPtr(new ArrayIndexExpression(k,
                MacroExpression(RK_COEF_GET_OFFSET,
                std::move(index),
                std::move(component))));
    }

    /**
     * @brief Get the RK coef for a given index/component
     * @details Expression rvalue version
     */
    template <class IndexExpression, class ComponentExpression>
    ExpressionPtr getK(IndexExpression&& index, ComponentExpression&& component) {
        return getK(
                ExpressionPtr(new IndexExpression(std::move(index))),
                ExpressionPtr(new ComponentExpression(std::move(component))));
    }

    EntryPointSig entry_ = nullptr;
    Poco::SharedLibrary so_;
    CudaGeneratorSBML sbmlgen_;
    GPUSimExecutableModel& mod_;
    // the RK coefficients
    Variable* k = nullptr;
    Macro* RK_COEF_GET_OFFSET = nullptr;
    const FunctionParameter* h = nullptr;
    Variable* rk_gen = nullptr;
    Variable* f = nullptr;
};

CudaGenerator::CudaGenerator() {

}

CudaGenerator::~CudaGenerator() {

}

void CudaGenerator::generate(GPUSimExecutableModel& model) {
    impl_.reset(new CudaGeneratorImpl(model));
    impl_->generate();
}

CudaGenerator::EntryPointSig CudaGenerator::getEntryPoint() {
    return impl_->getEntryPoint();
}

CudaGeneratorImpl::EntryPointSig CudaGeneratorImpl::getEntryPoint() {
    if (!entry_)
        throw_gpusim_exception("No entry point set (did you forget to call generate first?)");
    return entry_;
}

ExpressionPtr CudaGeneratorImpl::getVecRK(const FloatingSpecies* s) {
//     if (rk_index == 0) {
        return ExpressionPtr(new ArrayIndexExpression(f,
                    MacroExpression(RK_COEF_GET_OFFSET,
                    VariableRefExpression(rk_gen),
                    LiteralIntExpression(mod_.getStateVecComponent(s)))));
//     } else {
//         return ExpressionPtr(new SumExpression(ArrayIndexExpression(k,
//                     MacroExpression(RK_COEF_GET_OFFSET,
//                     LiteralIntExpression(rk_index),
//                     LiteralIntExpression(mod_.getStateVecComponent(s)))),
//                     ProductExpression(ProductExpression(RealLiteralExpression(0.5), VariableRefExpression(h)),
//                     getVecRK(s, rk_index-1))));
//     }
}

# if GPUSIM_MODEL_USE_SBML
ExpressionPtr CudaGeneratorImpl::generateExpForSBMLASTNode(const Reaction* r, const libsbml::ASTNode* node) {
    assert(r && node);
    switch (node->getType()) {
        case libsbml::AST_TIMES:
            return ExpressionPtr(new ProductExpression(generateExpForSBMLASTNode(r, node->getChild(0)), generateExpForSBMLASTNode(r, node->getChild(1))));
        case libsbml::AST_NAME:
            if (r->isParameter(node->getName()))
                return ExpressionPtr(new RealLiteralExpression(r->getParameterVal(node->getName())));
            if (r->isParticipant(node->getName()))
                return getVecRK(r->getSpecies(node->getName()));
            else
                return ExpressionPtr(new LiteralIntExpression(12345));
        default:
            return ExpressionPtr(new LiteralIntExpression(0));
    }
}
# else
# error "Need SBML"
# endif

ExpressionPtr CudaGeneratorImpl::generateReactionRateExp(const Reaction* r) {
# if GPUSIM_MODEL_USE_SBML
    return generateExpForSBMLASTNode(r, r->getSBMLMath());
# else
# error "Need SBML"
# endif
}

ExpressionPtr CudaGeneratorImpl::accumulate(ExpressionPtr&& sum, ExpressionPtr&& item, bool invert) {
    if (sum)
        return invert ?
            ExpressionPtr(new SubtractExpression(std::move(sum), std::move(item))) :
            ExpressionPtr(new SumExpression(std::move(sum), std::move(item)));
    else
        return invert ?
            ExpressionPtr(new UnaryMinusExpression(std::move(item))) :
            ExpressionPtr(std::move(item));
}

ExpressionPtr CudaGeneratorImpl::generateEvalExp(int component, int rk_index) {
    // get the species corresponding to this component in the state vec
    const FloatingSpecies* s = mod_.getFloatingSpeciesFromSVComponent(component);

    // accumulate reaction rates in this expression
    ExpressionPtr sum;

    for (const Reaction* r : mod_.getReactions())
        if (int factor = mod_.getReactionSideFac(r, s)) {
            assert((factor == 1 || factor == -1) && "Should not happen");
            sum = accumulate(std::move(sum), generateReactionRateExp(r), factor==-1);
        }

    // expand for RK coefficients
    if (rk_index == 0)
        return std::move(sum);
    else if (rk_index == 1 || rk_index == 2)
        return ExpressionPtr(
            new SumExpression(
                std::move(sum),
                ProductExpression(
                    ProductExpression(RealLiteralExpression(0.5),VariableRefExpression(h)),
                    getK(LiteralIntExpression(rk_index-1), LiteralIntExpression(component))
                  )));
    else if (rk_index == 3)
        return ExpressionPtr(
            new SumExpression(
                std::move(sum),
                ProductExpression(
                    VariableRefExpression(h),
                    getK(LiteralIntExpression(rk_index-1), LiteralIntExpression(component))
                  )));
    else
        assert(0 && "Should not happen (0 <= rk_index < 4)");
}

void CudaGeneratorImpl::generate() {
    auto generate_start = std::chrono::high_resolution_clock::now();

    CudaModule mod;
    std::string entryName = "cuEntryPoint";

    // get the size of the state vector
    int n = mod_.getStateVector(NULL);

    // construct the DOM

    // macros
    Macro* N = mod.addMacro(Macro("N", std::to_string(n)));

    Macro* RK4ORDER = mod.addMacro(Macro("RK4ORDER", "4"));

    Macro* RK_COEF_LEN = mod.addMacro(Macro("RK_COEF_LEN", "RK4ORDER*RK4ORDER*N"));
    RK_COEF_GET_OFFSET = mod.addMacro(Macro("RK_COEF_GET_OFFSET", "RK4ORDER*N + idx*N + component","idx", "component"));
//     Macro* RK_COEF_SIZE = mod.addMacro(Macro("RK_COEF_SIZE", ""));

//     Macro* RK_GET_INDEX = mod.addMacro(Macro("RK_GET_INDEX", "(threadIdx.x%N)"));

    Macro* RK_GET_COMPONENT = mod.addMacro(Macro("RK_GET_COMPONENT", "(threadIdx.x/32)"));

    Macro* RK_STATE_VEC_LEN = mod.addMacro(Macro("RK_STATE_VEC_LEN", "RK4ORDER*N"));
    Macro* RK_STATE_VEC_GET_OFFSET = mod.addMacro(Macro("RK_STATE_VEC_GET_OFFSET", "gen*N + component", "gen", "component"));

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
    h = kernel->getPositionalParam(0);
    assert(h->getType()->isIdentical(RKReal) && "h has wrong type - signature of kernel changed? ");

    // generate the kernel
    {
        // declare shared memory
//         kernel->addStatement(ExpressionPtr(new CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "k")), true)));
        Variable* shared_buf = CudaVariableDeclarationExpression::downcast(ExpressionStatement::insert(*kernel, CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "shared_buf")), true))->getExpression())->getVariable();

        k = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "k")), ReferenceExpression(ArrayIndexExpression(shared_buf, LiteralIntExpression(0)))))->getExpression())->getVariable();

        f = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "f")), ReferenceExpression(ArrayIndexExpression(k, MacroExpression(RK_COEF_LEN)))))->getExpression())->getVariable();

        Variable* t = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "t")), ReferenceExpression(ArrayIndexExpression(f, MacroExpression(RK_STATE_VEC_LEN)))))->getExpression())->getVariable();

        // printf
        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));

        {
            // initialize k (the RK coefficients)

            for (int i=0; i<4; ++i) {
                // init expression for k
                AssignmentExpression* k_init_assn =
                    AssignmentExpression::downcast(ExpressionStatement::insert(*kernel, AssignmentExpression(
                    getK(LiteralIntExpression(i), MacroExpression(RK_GET_COMPONENT)),
                    LiteralIntExpression(0)))->getExpression());

                // printf showing the init'd value of k
                ExpressionStatement::insert(*kernel, FunctionCallExpression(
                    mod.getPrintf(),
                    StringLiteralExpression("k[RK_COEF_GET_OFFSET(%d, %d)] = %f\\n"),
                    LiteralIntExpression(i),
                    MacroExpression(RK_GET_COMPONENT),
                    k_init_assn->getLHS()->clone()
                ));
            }
        }

        {
            SwitchStatement* component_switch = SwitchStatement::insert(*kernel, MacroExpression(RK_GET_COMPONENT));

            for (int component=0; component<n; ++component) {
                component_switch->addCase(LiteralIntExpression(component));

                // initialize f (the state vector)
                AssignmentExpression* f_init_assn =
                    AssignmentExpression::downcast(ExpressionStatement::insert(component_switch->getBody(),
                    AssignmentExpression(
                        ArrayIndexExpression(f,
                MacroExpression(RK_STATE_VEC_GET_OFFSET,
                LiteralIntExpression(0),
                LiteralIntExpression(component))),
                    RealLiteralExpression(mod_.getFloatingSpeciesFromSVComponent(component)->getInitialConcentration())))->getExpression());

                ExpressionStatement::insert(component_switch->getBody(), FunctionCallExpression(
                    mod.getPrintf(),
                    StringLiteralExpression("f[RK_STATE_VEC_GET_OFFSET(%d, %d)] = %f\\n"),
                    LiteralIntExpression(0),
                    LiteralIntExpression(component),
                    f_init_assn->getLHS()->clone()
                ));

                component_switch->addBreak();
            }
        }

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

        // main integration loop
        {
            ForStatement* update_coef_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

            rk_gen = update_coef_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

            update_coef_loop->setInitExp(ExpressionPtr(new VariableInitExpression(rk_gen, ExpressionPtr(new LiteralIntExpression(0)))));

            update_coef_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(rk_gen)), ExpressionPtr(new LiteralIntExpression(1)))));

            update_coef_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(rk_gen)))));

            SwitchStatement* component_switch = SwitchStatement::insert(update_coef_loop->getBody(), MacroExpression(RK_GET_COMPONENT));

            for (int component=0; component<n; ++component) {
                component_switch->addCase(LiteralIntExpression(component));

                for (int i=0; i<4; ++i) {
                    AssignmentExpression* k_assn =
                        AssignmentExpression::downcast(ExpressionStatement::insert(component_switch->getBody(), AssignmentExpression(
                        ExpressionPtr(new ArrayIndexExpression(k,
                        MacroExpression(RK_COEF_GET_OFFSET,
                        LiteralIntExpression(i),
                        LiteralIntExpression(component)))),
                        generateEvalExp(component, i)))->getExpression());
                }

                component_switch->addBreak();
            }

            update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));

            // update the state vector
            ExpressionStatement::insert(update_coef_loop->getBody(), AssignmentExpression(
                        ExpressionPtr(new ArrayIndexExpression(f,
                        MacroExpression(RK_STATE_VEC_GET_OFFSET,
                        SumExpression(VariableRefExpression(rk_gen), LiteralIntExpression(1)),
                        MacroExpression(RK_GET_COMPONENT)))),
                        SumExpression(
                        // prev value
                        ArrayIndexExpression(f,
                        MacroExpression(RK_STATE_VEC_GET_OFFSET,
                        VariableRefExpression(rk_gen),
                        MacroExpression(RK_GET_COMPONENT))),
                        // RK approx. for increment
                        ProductExpression(
                        ProductExpression(RealLiteralExpression(0.166666667), VariableRefExpression(h)),
                        SumExpression(SumExpression(SumExpression(
                            getK(LiteralIntExpression(0), MacroExpression(RK_GET_COMPONENT)),
                            ProductExpression(RealLiteralExpression(2.), getK(LiteralIntExpression(1), MacroExpression(RK_GET_COMPONENT)))),
                            ProductExpression(RealLiteralExpression(2.), getK(LiteralIntExpression(2), MacroExpression(RK_GET_COMPONENT)))),
                            getK(LiteralIntExpression(3), MacroExpression(RK_GET_COMPONENT))
                          )))
              ));
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

    auto generate_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Generating model DOM took " << std::chrono::duration_cast<std::chrono::milliseconds>(generate_finish - generate_start).count() << " ms";

    auto serialize_start = std::chrono::high_resolution_clock::now();

    // serialize the module to a document

    {
        Serializer s("/tmp/rr_cuda_model.cu");
        mod.serialize(s);
    }

    auto serialize_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Serializing model took " << std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish - serialize_start).count() << " ms";

    auto compile_start = std::chrono::high_resolution_clock::now();

    // compile the module

    FILE* pp = popen("nvcc -D__CUDACC__ -ccbin gcc -m32 -I/home/jkm/devel/src/roadrunner/source --ptxas-options=-v --compiler-options '-fPIC' -Drr_cuda_model_EXPORTS -Xcompiler ,\"-fPIC\",\"-fPIC\",\"-g\" -DNVCC --shared -o /tmp/rr_cuda_model.so /tmp/rr_cuda_model.cu 2>&1 >/dev/null", "r");

#define SBUFLEN 512
    char sbuf[SBUFLEN];
    fgets(sbuf, SBUFLEN, pp);

    int code = pclose(pp);
    pp = NULL;
    // 512 for warning
    Log(Logger::LOG_DEBUG) << "nvcc return code: " << code << "\n";

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

    auto compile_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Compiling model took " << std::chrono::duration_cast<std::chrono::milliseconds>(compile_finish - compile_start).count() << " ms";

    auto load_start = std::chrono::high_resolution_clock::now();

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

    auto load_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Loading model took " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish - load_start).count() << " ms";

}

} // namespace dom

} // namespace rrgpu

} // namespace rr

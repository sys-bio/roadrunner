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
# include "Hasher.hpp"
# include "rrUtils.h"
# include "GitInfo.h"

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
      : mod_(mod) {
        if (Logger::getLevel() >= Logger::LOG_TRACE)
            enableDiagnostics();
    }

    ExpressionPtr generateReactionRateExp(const Reaction* r, int rk_index);

    ExpressionPtr accumulate(ExpressionPtr&& sum, ExpressionPtr&& item, bool invert);

    ExpressionPtr generateEvalExp(int component, int rk_index);

    ExpressionPtr getVecRK(const FloatingSpecies* s, int rk_index);

# if 0 && GPUSIM_MODEL_USE_SBML
    ExpressionPtr generateExpForSBMLASTNode(const Reaction* r, const libsbml::ASTNode* node, int rk_index);
// # else
// # error "Need SBML"
# endif

    ExpressionPtr generateExpForASTNode(const ModelASTNode* n, int rk_index);

    /// Generate the model code
    void generate();

    /// Entry point into generated code
    EntryPointSig getEntryPoint();

    /// Return true if diagnostics should be emitted from the GPU code
    bool diagnosticsEnabled() const {
        return diag_;
    }

    /// Enable diagnostics dumped directly from the GPU
    void enableDiagnostics() {
        diag_ = true;
    }

protected:

    ExpressionPtr getRKCoef(const Variable* coef, ExpressionPtr&& index, ExpressionPtr&& component) {
        return ExpressionPtr(new ArrayIndexExpression(coef,
                MacroExpression(RK_COEF_GET_OFFSET,
                std::move(index),
                std::move(component))));
    }

    ExpressionPtr getK(ExpressionPtr&& index, ExpressionPtr&& component) {
        return getRKCoef(k, std::move(index), std::move(component));
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

    ExpressionPtr getVecCoef(const Variable* vec, ExpressionPtr&& generation, ExpressionPtr&& component) {
        return ExpressionPtr(new ArrayIndexExpression(vec,
                MacroExpression(RK_STATE_VEC_GET_OFFSET,
                std::move(generation),
                std::move(component)
                  )));
    }

    ExpressionPtr getSV(ExpressionPtr&& generation, ExpressionPtr&& component) {
        return getVecCoef(f, std::move(generation), std::move(component));
    }

    /**
     * @brief Get the RK coef for a given index/component
     * @details Expression rvalue version
     */
    template <class GenerationExpression, class ComponentExpression>
    ExpressionPtr getSV(GenerationExpression&& generation, ComponentExpression&& component) {
        return getSV(
                ExpressionPtr(new GenerationExpression(std::move(generation))),
                ExpressionPtr(new ComponentExpression(std::move(component))));
    }

    EntryPointSig entry_ = nullptr;
    Poco::SharedLibrary so_;
    CudaGeneratorSBML sbmlgen_;
    GPUSimExecutableModel& mod_;
    CudaModule mod;

    // the RK coefficients
    Macro* RK_COEF_GET_OFFSET = nullptr;
    Macro* RK_STATE_VEC_GET_OFFSET = nullptr;

    Variable* h = nullptr;
    Variable* rk_gen = nullptr;
    Variable* k = nullptr;
    Variable* f = nullptr;
    bool diag_ = false;
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

ExpressionPtr CudaGeneratorImpl::getVecRK(const FloatingSpecies* s, int rk_index) {
    assert(s && "Should not happen");
    int component = mod_.getStateVecComponent(s);
    if (rk_index == 0)
        return getSV(VariableRefExpression(rk_gen), LiteralIntExpression(component));
    else if (rk_index == 1 || rk_index == 2)
        return ExpressionPtr(new SumExpression(getSV(VariableRefExpression(rk_gen), LiteralIntExpression(component)),
                    ProductExpression(
                        ProductExpression(RealLiteralExpression(0.5), VariableRefExpression(h)),
                        getK(LiteralIntExpression(rk_index-1), LiteralIntExpression(component))
                      )));
    else if (rk_index == 3)
        return ExpressionPtr(new SumExpression(getSV(VariableRefExpression(rk_gen), LiteralIntExpression(component)),
                    ProductExpression(
                        VariableRefExpression(h),
                        getK(LiteralIntExpression(rk_index-1), LiteralIntExpression(component))
                      )));
    else
        assert(0 && "Should not happen (0 <= rk_index < 4)");
}

ExpressionPtr CudaGeneratorImpl::generateExpForASTNode(const ModelASTNode* node, int rk_index) {
    assert(node);
    // visitor is no good here (way too much overhead and not extensible enough)
    // sum
    if (auto n = dynamic_cast<const SumASTNode*>(node))
        return ExpressionPtr(new SumExpression(generateExpForASTNode(n->getLeft(), rk_index), generateExpForASTNode(n->getRight(), rk_index)));
    // difference
    else if (auto n = dynamic_cast<const DifferenceASTNode*>(node))
        return ExpressionPtr(new DifferenceExpression(generateExpForASTNode(n->getLeft(), rk_index), generateExpForASTNode(n->getRight(), rk_index)));
    // product
    else if (auto n = dynamic_cast<const ProductASTNode*>(node))
        return ExpressionPtr(new ProductExpression(generateExpForASTNode(n->getLeft(), rk_index), generateExpForASTNode(n->getRight(), rk_index)));
    // division
    else if (auto n = dynamic_cast<const QuotientASTNode*>(node))
        return ExpressionPtr(new QuotientExpression(generateExpForASTNode(n->getLeft(), rk_index), generateExpForASTNode(n->getRight(), rk_index)));
    // exponentiation
    else if (auto n = dynamic_cast<const ExponentiationASTNode*>(node))
        return ExpressionPtr(mod.pow(generateExpForASTNode(n->getLeft(), rk_index), generateExpForASTNode(n->getRight(), rk_index)));
    // floating species ref
    else if (auto n = dynamic_cast<const FloatingSpeciesRefASTNode*>(node))
        return getVecRK(n->getFloatingSpecies(), rk_index);
    // parameter ref
    else if (auto n = dynamic_cast<const ParameterRefASTNode*>(node))
        return ExpressionPtr(new RealLiteralExpression((n->getParameterVal(), rk_index)));
    // integer
    else if (auto n = dynamic_cast<const IntegerLiteralASTNode*>(node))
        return ExpressionPtr(new LiteralIntExpression(n->getValue()));
    else
        return ExpressionPtr(new LiteralIntExpression(12345));
}

# if 0 && GPUSIM_MODEL_USE_SBML
ExpressionPtr CudaGeneratorImpl::generateExpForSBMLASTNode(const Reaction* r, const libsbml::ASTNode* node, int rk_index) {
    assert(r && node);
    switch (node->getType()) {
        case libsbml::AST_TIMES:
            return ExpressionPtr(new ProductExpression(generateExpForSBMLASTNode(r, node->getChild(0), rk_index), generateExpForSBMLASTNode(r, node->getChild(1), rk_index)));
        case libsbml::AST_NAME:
            if (r->isParameter(node->getName()))
                return ExpressionPtr(new RealLiteralExpression(r->getParameterVal(node->getName())));
            if (r->isParticipant(node->getName()))
                return getVecRK(r->getFloatingSpecies(node->getName()), rk_index);
            else
                return ExpressionPtr(new LiteralIntExpression(12345));
        default:
            return ExpressionPtr(new LiteralIntExpression(0));
    }
}
// # else
// # error "Need SBML"
# endif

ExpressionPtr CudaGeneratorImpl::generateReactionRateExp(const Reaction* r, int rk_index) {
    return generateExpForASTNode(r->getKineticLaw().getAlgebra().getRoot(), rk_index);
}

ExpressionPtr CudaGeneratorImpl::accumulate(ExpressionPtr&& sum, ExpressionPtr&& item, bool invert) {
    if (sum)
        return invert ?
            ExpressionPtr(new DifferenceExpression(std::move(sum), std::move(item))) :
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
            sum = accumulate(std::move(sum), generateReactionRateExp(r, rk_index), factor==-1);
        }

    return std::move(sum);
}

void CudaGeneratorImpl::generate() {
    auto generate_start = std::chrono::high_resolution_clock::now();

    std::string entryName = "cuEntryPoint";

    // dump the state vector assignments
    mod_.dumpStateVecAssignments();

    // get the size of the state vector
    int n = mod_.getStateVector(NULL);

    // construct the DOM

    // macros
    Macro* N = mod.addMacro(Macro("N", std::to_string(n)));

    Macro* RK4ORDER = mod.addMacro(Macro("RK4ORDER", "4"));

    Macro* RK_COEF_LEN = mod.addMacro(Macro("RK_COEF_LEN", "(RK4ORDER*N)"));
    RK_COEF_GET_OFFSET = mod.addMacro(Macro("RK_COEF_GET_OFFSET", "((idx)*N + (component))","idx", "component"));

    Macro* RK_GET_COMPONENT = mod.addMacro(Macro("RK_GET_COMPONENT", "(threadIdx.x/32)"));

    Macro* RK_STATE_VEC_LEN = mod.addMacro(Macro("RK_STATE_VEC_LEN", "(km*N)"));
    RK_STATE_VEC_GET_OFFSET = mod.addMacro(Macro("RK_STATE_VEC_GET_OFFSET", "((gen)*N + (component))", "gen", "component"));

    Macro* RK_TIME_VEC_LEN = mod.addMacro(Macro("RK_TIME_VEC_LEN", "km"));

    // typedef for float or double
    Type* RKReal = TypedefStatement::downcast(mod.addStatement(StatementPtr(new TypedefStatement(BaseTypes::getTp(BaseTypes::FLOAT), "RKReal"))))->getAlias();

    Type* pRKReal = BaseTypes::get().addPointer(RKReal);

    Type* pRKReal_volatile = BaseTypes::get().addVolatile(pRKReal);

    CudaFunction* PrintCoefs = mod.addFunction(CudaFunction("PrintCoefs", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(pRKReal, "k")}));

    {
        PrintCoefs->setIsDeviceFun(true);

        PrintCoefs->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));

        IfStatement* if_thd1 = IfStatement::downcast(PrintCoefs->addStatement(StatementPtr(new IfStatement(
                    ExpressionPtr(new EqualityCompExpression(mod.getThreadIdx(CudaModule::IndexComponent::x), LiteralIntExpression(0)))
                ))));

        // printf
        if_thd1->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("PrintCoefs\\n")))));

        std::string printcoef_fmt_str;
        for (int rk_index=0; rk_index<4; ++rk_index) {
            for (int component=0; component<n; ++component) {
                ExpressionStatement::insert(if_thd1->getBody(), FunctionCallExpression(
                        mod.getPrintf(),
                        StringLiteralExpression("%3.3f "),
                        getRKCoef(PrintCoefs->getPositionalParam(0), ExpressionPtr(new LiteralIntExpression(rk_index)), ExpressionPtr(new LiteralIntExpression(component)))
                    ));
            }
            ExpressionStatement::insert(if_thd1->getBody(), FunctionCallExpression(
                mod.getPrintf(),
                StringLiteralExpression("\\n")
                ));
        }

        PrintCoefs->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
    }

    CudaFunction* PrintStatevec = mod.addFunction(CudaFunction("PrintStatevec", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(pRKReal, "f"), FunctionParameter(BaseTypes::getTp(BaseTypes::INT), "generation")}));

    {
        PrintStatevec->setIsDeviceFun(true);

        PrintStatevec->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));

        IfStatement* if_thd1 = IfStatement::downcast(PrintStatevec->addStatement(StatementPtr(new IfStatement(
            ExpressionPtr(new EqualityCompExpression(mod.getThreadIdx(CudaModule::IndexComponent::x), LiteralIntExpression(0)))
            ))));

        std::string sv_fmt_str = "statevec ";
        for (int component=0; component<n; ++component)
            sv_fmt_str += "%f ";
        sv_fmt_str += "\\n";

        // print the state vector
        FunctionCallExpression* printf_statevec = FunctionCallExpression::downcast(ExpressionStatement::insert(if_thd1->getBody(), FunctionCallExpression(
                mod.getPrintf(),
                StringLiteralExpression(sv_fmt_str)
            ))->getExpression());

        for (int component=0; component<n; ++component)
            printf_statevec->passArgument(ExpressionPtr(
                new ArrayIndexExpression(PrintStatevec->getPositionalParam(0),
                    MacroExpression(RK_STATE_VEC_GET_OFFSET,
                    VariableRefExpression(PrintStatevec->getPositionalParam(1)),
                    LiteralIntExpression(component)))
            ));

        PrintStatevec->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
    }

    CudaKernel* kernel = CudaKernel::downcast(mod.addFunction(CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(BaseTypes::getTp(BaseTypes::INT), "km"), FunctionParameter(pRKReal, "kt"), FunctionParameter(pRKReal, "kv")})));

    // the step size
    assert(kernel->getNumPositionalParams() == 3 && "wrong num of params - signature of kernel changed?");
    assert(kernel->getPositionalParam(0)->getType()->isIdentical(BaseTypes::getTp(BaseTypes::INT)) && "param has wrong type - signature of kernel changed? ");
    assert(kernel->getPositionalParam(1)->getType()->isIdentical(pRKReal) && "param has wrong type - signature of kernel changed? ");
    assert(kernel->getPositionalParam(2)->getType()->isIdentical(pRKReal) && "param has wrong type - signature of kernel changed? ");

    // generate the kernel
    {
        const FunctionParameter* km = kernel->getPositionalParam(0);
        const FunctionParameter* kt = kernel->getPositionalParam(1);
        const FunctionParameter* kv = kernel->getPositionalParam(2);

        // declare shared memory
        Variable* shared_buf = CudaVariableDeclarationExpression::downcast(ExpressionStatement::insert(*kernel, CudaVariableDeclarationExpression(kernel->addVariable(Variable(BaseTypes::get().addArray(RKReal), "shared_buf")), true))->getExpression())->getVariable();

        k = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "k")), ReferenceExpression(ArrayIndexExpression(shared_buf, LiteralIntExpression(0)))))->getExpression())->getVariable();

        f = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "f")), ReferenceExpression(ArrayIndexExpression(k, MacroExpression(RK_COEF_LEN)))))->getExpression())->getVariable();

        Variable* t = VariableInitExpression::downcast(ExpressionStatement::insert(*kernel, VariableInitExpression(kernel->addVariable(Variable(pRKReal, "t")), ReferenceExpression(ArrayIndexExpression(f, MacroExpression(RK_STATE_VEC_LEN)))))->getExpression())->getVariable();

//         if (diagnosticsEnabled()) {
//             // printf
//             kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));
//         }

        {
            // initialize k (the RK coefficients)

            for (int i=0; i<4; ++i) {
                // init expression for k
                AssignmentExpression* k_init_assn =
                    AssignmentExpression::downcast(ExpressionStatement::insert(*kernel, AssignmentExpression(
                    getK(LiteralIntExpression(i), MacroExpression(RK_GET_COMPONENT)),
                    LiteralIntExpression(0)))->getExpression());

//                 if (diagnosticsEnabled()) {
//                     // printf showing the init'd value of k
//                     ExpressionStatement::insert(*kernel, FunctionCallExpression(
//                         mod.getPrintf(),
//                         StringLiteralExpression("k[RK_COEF_GET_OFFSET(%d, %d)] = %f\\n"),
//                         LiteralIntExpression(i),
//                         MacroExpression(RK_GET_COMPONENT),
//                         k_init_assn->getLHS()->clone()
//                     ));
//                 }
            }


            // sync & print coefs

            IfStatement* if_thd1 = IfStatement::downcast(kernel->addStatement(StatementPtr(new IfStatement(
                    ExpressionPtr(new EqualityCompExpression(mod.getThreadIdx(CudaModule::IndexComponent::x), LiteralIntExpression(0)))
                ))));
            // printf
            if_thd1->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("**** Initial rk coefs ****\\n")))));

            kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
            kernel->addStatement(ExpressionPtr(new FunctionCallExpression(PrintCoefs, VariableRefExpression(k))));

//             for (int i=0; i<4; ++i) {
//                 if (diagnosticsEnabled()) {
//                     // printf showing the init'd value of k
//                     ExpressionStatement::insert(*kernel, FunctionCallExpression(
//                         mod.getPrintf(),
//                         StringLiteralExpression("k[RK_COEF_GET_OFFSET(%d, %d)] = %f\\n"),
//                         LiteralIntExpression(i),
//                         MacroExpression(RK_GET_COMPONENT),
//                         getK(LiteralIntExpression(i), MacroExpression(RK_GET_COMPONENT))
//                     ));
//                 }
//             }
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

//                 if (diagnosticsEnabled()) {
//                     ExpressionStatement::insert(component_switch->getBody(), FunctionCallExpression(
//                         mod.getPrintf(),
//                         StringLiteralExpression("f[RK_STATE_VEC_GET_OFFSET(%d, %d)] = %f\\n"),
//                         LiteralIntExpression(0),
//                         LiteralIntExpression(component),
//                         f_init_assn->getLHS()->clone()
//                     ));
//                 }

                component_switch->addBreak();
            }
        }

        // initialize the time values
        kernel->addStatement(ExpressionPtr(
            new FunctionCallExpression(
                mod.getRegMemcpy(),
                VariableRefExpression(t),
                VariableRefExpression(kt),
                ProductExpression(
                    VariableRefExpression(km), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))
                  )
              )));

        kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));

        if (diagnosticsEnabled()) {
            // print coefs
            // BUG? order reversed?
            kernel->addStatement(ExpressionPtr(new FunctionCallExpression(PrintCoefs, VariableRefExpression(k))));
            kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
        }

        // main integration loop
        {
            ForStatement* update_coef_loop = ForStatement::downcast(kernel->addStatement(ForStatement::make()));

            rk_gen = update_coef_loop->addVariable(Variable(BaseTypes::getTp(BaseTypes::INT), "j"));

            update_coef_loop->setInitExp(ExpressionPtr(new VariableInitExpression(rk_gen, ExpressionPtr(new LiteralIntExpression(0)))));

            update_coef_loop->setCondExp(ExpressionPtr(new LTComparisonExpression(ExpressionPtr(new VariableRefExpression(rk_gen)), ExpressionPtr(
                new DifferenceExpression(VariableRefExpression(km), LiteralIntExpression(1))
              ))));

            update_coef_loop->setLoopExp(ExpressionPtr(new PreincrementExpression(ExpressionPtr(new VariableRefExpression(rk_gen)))));

            // get step size
            h = update_coef_loop->addVariable(Variable(RKReal, "h"));
            VariableInitExpression::downcast(ExpressionStatement::insert(update_coef_loop->getBody(),
            VariableInitExpression(h,
              DifferenceExpression(
                  ArrayIndexExpression(t, SumExpression(VariableRefExpression(rk_gen), LiteralIntExpression(1))),
                  ArrayIndexExpression(t, VariableRefExpression(rk_gen))
              )
              ))->getExpression());

            for (int rk_index=0; rk_index<4; ++rk_index) {
                SwitchStatement* component_switch = SwitchStatement::insert(update_coef_loop->getBody(), MacroExpression(RK_GET_COMPONENT));

                for (int component=0; component<n; ++component) {
                    component_switch->addCase(LiteralIntExpression(component));

                // RK coef computation
                AssignmentExpression* k_assn =
                    AssignmentExpression::downcast(ExpressionStatement::insert(component_switch->getBody(), AssignmentExpression(
                    ExpressionPtr(new ArrayIndexExpression(k,
                    MacroExpression(RK_COEF_GET_OFFSET,
                    LiteralIntExpression(rk_index),
                    LiteralIntExpression(component)))),
                    generateEvalExp(component, rk_index)))->getExpression());

                    component_switch->addBreak();
                }

                update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
            }

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

            update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));

            if (diagnosticsEnabled()) {
                update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(PrintStatevec, VariableRefExpression(f), SumExpression(VariableRefExpression(rk_gen), LiteralIntExpression(1)))));

//                     ExpressionStatement::insert(if_thd1->getBody(), FunctionCallExpression(
//                             mod.getPrintf(),
//                             StringLiteralExpression("h = %f\\n"),
//                             VariableRefExpression(h)
//                         ));
                update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
            }
            if (diagnosticsEnabled()) {
                // print coefs
                update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaSyncThreads())));
                update_coef_loop->getBody().addStatement(ExpressionPtr(new FunctionCallExpression(PrintCoefs, VariableRefExpression(k))));
            }
        }

        // copy results to passed parameters
        kernel->addStatement(ExpressionPtr(
            new FunctionCallExpression(
                mod.getRegMemcpy(),
                VariableRefExpression(kv),
                VariableRefExpression(f),
                ProductExpression(ProductExpression(VariableRefExpression(km), MacroExpression(N)), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal)))
              )));
    }

    CudaFunction* entry = mod.addFunction(CudaFunction(entryName, BaseTypes::getTp(BaseTypes::VOID), {FunctionParameter(BaseTypes::getTp(BaseTypes::INT), "km"), FunctionParameter(pRKReal, "kt"), FunctionParameter(pRKReal, "kv")}));

    // generate the entry
    {
        const FunctionParameter* km = entry->getPositionalParam(0);
        const FunctionParameter* kt = entry->getPositionalParam(1);
        const FunctionParameter* kv = entry->getPositionalParam(2);

        if (diagnosticsEnabled()) {
            // printf
            entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in cuda\\n")))));
        }

        // allocate mem for tvalues
        Variable* tvals =
            VariableDeclarationExpression::downcast(ExpressionStatement::insert(entry,
            VariableDeclarationExpression(entry->addVariable(Variable(pRKReal, "tvals"))))->getExpression())->getVariable();
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaMalloc(),
        ReferenceExpression(VariableRefExpression(tvals)),
        ProductExpression(VariableRefExpression(km), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))))));

        // copy tvalues to device memory
        entry->addStatement(ExpressionPtr(
            new FunctionCallExpression(
                mod.getCudaMemcpy(),
                VariableRefExpression(tvals),
                VariableRefExpression(kt),
                ProductExpression(VariableRefExpression(km), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))),
                SymbolExpression("cudaMemcpyHostToDevice")
              )));

        // allocate mem for results
        Variable* results =
            VariableDeclarationExpression::downcast(ExpressionStatement::insert(entry,
            VariableDeclarationExpression(entry->addVariable(Variable(pRKReal, "results"))))->getExpression())->getVariable();
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaMalloc(),
        ReferenceExpression(VariableRefExpression(results)),
        ProductExpression(ProductExpression(VariableRefExpression(km), MacroExpression(N)), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))))));

        // call the kernel
        CudaKernelCallExpression* kernel_call = CudaKernelCallExpression::downcast(ExpressionStatement::insert(entry,
        CudaKernelCallExpression(1, 1, 1,
        kernel,
        VariableRefExpression(km),
        VariableRefExpression(tvals),
        VariableRefExpression(results)
          ))->getExpression());

        kernel_call->setNumThreads(ProductExpression(MacroExpression(N), LiteralIntExpression(32)));

        kernel_call->setSharedMemSize(SumExpression(SumExpression(
            ProductExpression(MacroExpression(RK_COEF_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))), // Coefficient size
            ProductExpression(MacroExpression(RK_STATE_VEC_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal)))), // State vector size
            ProductExpression(MacroExpression(RK_TIME_VEC_LEN), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))) // Time vector size
            ));

        // copy device results to passed parameters
        entry->addStatement(ExpressionPtr(
            new FunctionCallExpression(
                mod.getCudaMemcpy(),
                VariableRefExpression(kv),
                VariableRefExpression(results),
                ProductExpression(ProductExpression(VariableRefExpression(km), MacroExpression(N)), FunctionCallExpression(mod.getSizeof(), TypeRefExpression(RKReal))),
                SymbolExpression("cudaMemcpyDeviceToHost")
              )));

        // free the result mem
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaFree(), VariableRefExpression(results))));

        // call cudaDeviceSynchronize
        entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaDeviceSynchronize())));
    }

    auto generate_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Generating model DOM took " << std::chrono::duration_cast<std::chrono::milliseconds>(generate_finish - generate_start).count() << " ms";

    // generate temporary file names

//     Log(Logger::LOG_DEBUG) << "Last git commit: " << getGitLastCommit();

    // The git version info definitely doesn't work as of 4ad273339b84e3149196b5a08936375e42edc13b
    // (reports last commit 19abcd7d7de27be773c640bfdde6455c9bcf4125)
    std::string hashedid = mod_.getSBMLHash().str();

//     Log(Logger::LOG_DEBUG) << "Hashed model id: " << hashedid;

    std::string cuda_src_name = joinPath(getTempDir(), "rr_cuda_model_" + hashedid + ".cu");

    Log(Logger::LOG_DEBUG) << "CUDA source: " << cuda_src_name;

    std::string libname = joinPath(getTempDir(), "rr_cuda_model_" + hashedid + ".so");

    Log(Logger::LOG_DEBUG) << "CUDA object: " << libname;

    auto serialize_start = std::chrono::high_resolution_clock::now();

    // serialize the module to a document

    {
        Serializer s(cuda_src_name);
        mod.serialize(s);
    }

    auto serialize_finish = std::chrono::high_resolution_clock::now();

    Log(Logger::LOG_INFORMATION) << "Serializing model took " << std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish - serialize_start).count() << " ms";

    auto compile_start = std::chrono::high_resolution_clock::now();

    // compile the module

    std::string popenline = "nvcc -D__CUDACC__ -ccbin gcc -m32 --ptxas-options=-v --compiler-options '-fPIC' -Drr_cuda_model_EXPORTS -Xcompiler ,\"-fPIC\",\"-fPIC\",\"-g\" -DNVCC --shared -o "
      + libname + " "
      + cuda_src_name + " 2>&1 >/dev/null";

    Log(Logger::LOG_DEBUG) << "CUDA compiler line: " << popenline;

    FILE* pp = popen(popenline.c_str(), "r");

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

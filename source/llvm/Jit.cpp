//
// Created by Ciaran on 25/10/2021.
//

#include <stdint.h>
#include <unordered_map>
#include <rrRoadRunnerOptions.h>
#include "Jit.h"
#include "ModelResources.h"
#include "rrRoadRunnerOptions.h"
#include "SBMLModelObjectCache.h"
#include "rrRoadRunnerOptions.h"
#include "SBMLSupportFunctions.h"

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Host.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

using namespace rr;

namespace rrllvm {

    Jit::Jit(std::uint32_t options)
            : options(options),
              context(std::make_unique<llvm::LLVMContext>()),
            // todo the module name should be the sbmlMD5. Might be cleaner to
            //  add this as a parameter to Jit constructor.
              module(std::make_unique<llvm::Module>("LLVM Module", *context)),
              moduleNonOwning(module.get()), /*Maintain a weak ref so we don't lose our handle to the module*/
              builder(std::make_unique<llvm::IRBuilder<>>(*context)) {


        // IR module is initialized with just a ModuleID and a source filename
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        if (!context) {
            std::string err = "Context pointer is null";
            rrLogErr << err;
            throw_llvm_exception(err)
        }
        if (!module) {
            std::string err = "Module pointer is null";
            rrLogErr << err;
            throw_llvm_exception(err)
        }
        if (!builder) {
            std::string err = "builder pointer is null";
            rrLogErr << err;
            throw_llvm_exception(err)
        }

        createCLibraryFunctions();
    }

    Jit::Jit()
            : Jit(LoadSBMLOptions().modelGeneratorOpt) {}

    llvm::Module *Jit::getModuleNonOwning() {
        return moduleNonOwning;
    }

    llvm::LLVMContext *Jit::getContextNonOwning() {
        return context.get();
    }

    void Jit::setModuleIdentifier(const std::string &id) {
        moduleNonOwning->setModuleIdentifier(id);
    }

    std::string Jit::mangleName(const std::string &unmangledName) const {
        std::string mangledName;
        llvm::raw_string_ostream mangledNameStream(mangledName);
        llvm::Mangler::getNameWithPrefix(mangledNameStream, unmangledName, getDataLayout());
        return mangledNameStream.str();
    }

    llvm::IRBuilder<> *Jit::getBuilderNonOwning() {
        return builder.get();
    }

    void Jit::transferObjectsToResources(std::shared_ptr<rrllvm::ModelResources> modelResources) {
        modelResources->context = std::move(context);
        context = nullptr;
    }


    void Jit::mapLLVMGeneratedFunctionsToSymbols(ModelResources *modelResources, std::uint32_t options) {

        modelResources->evalInitialConditionsPtr = reinterpret_cast<EvalInitialConditionsCodeGen::FunctionPtr>(lookupFunctionAddress(
                "evalInitialConditions"));
        modelResources->evalReactionRatesPtr = reinterpret_cast<EvalReactionRatesCodeGen::FunctionPtr>(lookupFunctionAddress(
                "evalReactionRates"));
        modelResources->getBoundarySpeciesAmountPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getBoundarySpeciesAmount"));
        modelResources->getFloatingSpeciesAmountPtr = reinterpret_cast<GetFloatingSpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getFloatingSpeciesAmount"));
        modelResources->getBoundarySpeciesConcentrationPtr = reinterpret_cast<GetBoundarySpeciesConcentrationCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getBoundarySpeciesConcentration"));
        modelResources->getFloatingSpeciesConcentrationPtr = reinterpret_cast<GetFloatingSpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getFloatingSpeciesConcentration"));
        modelResources->getCompartmentVolumePtr = reinterpret_cast<GetCompartmentVolumeCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getCompartmentVolume"));
        modelResources->getGlobalParameterPtr = reinterpret_cast<GetGlobalParameterCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getGlobalParameter"));
        modelResources->evalRateRuleRatesPtr = reinterpret_cast<EvalRateRuleRatesCodeGen::FunctionPtr>(lookupFunctionAddress(
                "evalRateRuleRates"));
        modelResources->getEventTriggerPtr = reinterpret_cast<GetEventTriggerCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getEventTrigger"));
        modelResources->getEventPriorityPtr = reinterpret_cast<GetEventPriorityCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getEventPriority"));
        modelResources->getEventDelayPtr = reinterpret_cast<GetEventDelayCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getEventDelay"));
        modelResources->eventTriggerPtr = reinterpret_cast<EventTriggerCodeGen::FunctionPtr>(lookupFunctionAddress(
                "eventTrigger"));
        modelResources->eventAssignPtr = reinterpret_cast<EventAssignCodeGen::FunctionPtr>(lookupFunctionAddress(
                "eventAssign"));
        modelResources->getPiecewiseTriggerPtr = reinterpret_cast<GetPiecewiseTriggerCodeGen::FunctionPtr>(lookupFunctionAddress(
                "getPiecewiseTrigger"));
        modelResources->evalVolatileStoichPtr = reinterpret_cast<EvalVolatileStoichCodeGen::FunctionPtr>(lookupFunctionAddress(
                "evalVolatileStoich"));
        modelResources->evalConversionFactorPtr = reinterpret_cast<EvalConversionFactorCodeGen::FunctionPtr>(lookupFunctionAddress(
                "evalConversionFactor"));
        if (options & LoadSBMLOptions::READ_ONLY) {
            modelResources->setBoundarySpeciesAmountPtr = 0;
            modelResources->setBoundarySpeciesConcentrationPtr = 0;
            modelResources->setFloatingSpeciesConcentrationPtr = 0;
            modelResources->setCompartmentVolumePtr = 0;
            modelResources->setFloatingSpeciesAmountPtr = 0;
            modelResources->setGlobalParameterPtr = 0;
        } else {
            modelResources->setBoundarySpeciesAmountPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setBoundarySpeciesAmount"));
            modelResources->setBoundarySpeciesConcentrationPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setBoundarySpeciesConcentration"));
            modelResources->setFloatingSpeciesConcentrationPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setFloatingSpeciesConcentration"));
            modelResources->setCompartmentVolumePtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setCompartmentVolume"));
            modelResources->setBoundarySpeciesAmountPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setBoundarySpeciesAmount"));
            modelResources->setFloatingSpeciesAmountPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setFloatingSpeciesAmount"));
            modelResources->setGlobalParameterPtr = reinterpret_cast<SetGlobalParameterCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setGlobalParameter"));
        }

        if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS) {
            modelResources->getFloatingSpeciesInitConcentrationsPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getFloatingSpeciesInitConcentrations"));
            modelResources->setFloatingSpeciesInitConcentrationsPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setFloatingSpeciesInitConcentrations"));
            modelResources->getFloatingSpeciesInitAmountsPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getFloatingSpeciesInitAmounts"));
            modelResources->setFloatingSpeciesInitAmountsPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setFloatingSpeciesInitAmounts"));
            modelResources->getBoundarySpeciesInitConcentrationsPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getBoundarySpeciesInitConcentrations"));
            modelResources->setBoundarySpeciesInitConcentrationsPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setBoundarySpeciesInitConcentrations"));
            modelResources->getBoundarySpeciesInitAmountsPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getBoundarySpeciesInitAmounts"));
            modelResources->setBoundarySpeciesInitAmountsPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setBoundarySpeciesInitAmounts"));
            modelResources->getCompartmentInitVolumesPtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getCompartmentInitVolumes"));
            modelResources->setCompartmentInitVolumesPtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setCompartmentInitVolumes"));
            modelResources->getGlobalParameterInitValuePtr = reinterpret_cast<GetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "getGlobalParameterInitValue"));
            modelResources->setGlobalParameterInitValuePtr = reinterpret_cast<SetBoundarySpeciesAmountCodeGen::FunctionPtr>(lookupFunctionAddress(
                    "setGlobalParameterInitValue"));
        } else {
            modelResources->getFloatingSpeciesInitConcentrationsPtr = 0;
            modelResources->setFloatingSpeciesInitConcentrationsPtr = 0;

            modelResources->getFloatingSpeciesInitAmountsPtr = 0;
            modelResources->setFloatingSpeciesInitAmountsPtr = 0;

            modelResources->getBoundarySpeciesInitConcentrationsPtr = 0;
            modelResources->setBoundarySpeciesInitConcentrationsPtr = 0;

            modelResources->getBoundarySpeciesInitAmountsPtr = 0;
            modelResources->setBoundarySpeciesInitAmountsPtr = 0;

            modelResources->getCompartmentInitVolumesPtr = 0;
            modelResources->setCompartmentInitVolumesPtr = 0;

            modelResources->getGlobalParameterInitValuePtr = 0;
            modelResources->setGlobalParameterInitValuePtr = 0;
        }
    }

    std::string Jit::emitToString() {
        std::string str;
        llvm::raw_string_ostream os(str);
        os << *getModuleNonOwning();
        os.flush();
        return str;
    }

    void Jit::createCLibraryFunction(llvm::LibFunc funcId, llvm::FunctionType *funcType) {
        // For some reason I had a problem when I passed the default ctor for TargetLibraryInfoImpl in
        // std::string error;
        //TargetLibraryInfoImpl defaultImpl(TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), error));
        llvm::TargetLibraryInfoImpl defaultImpl;
        llvm::TargetLibraryInfo targetLib(defaultImpl);

        if (targetLib.has(funcId)) {
            const llvm::StringRef &name = targetLib.getName(funcId);
            llvm::Function::Create(
                    funcType, llvm::Function::ExternalLinkage, name, getModuleNonOwning()
            );
        } else {
            std::string msg = "native target does not have library function for ";
            msg += targetLib.getName(funcId);
            throw_llvm_exception(msg);
        }
    }

    void Jit::createCLibraryFunctions() {
        using namespace llvm;
//        LLVMContext &context = module->getContext();
        Type *double_type = Type::getDoubleTy(*context);
        Type *args_d1[] = {double_type};
        Type *args_d2[] = {double_type, double_type};

        /// double pow(double x, double y);
        createCLibraryFunction(LibFunc_pow,
                               FunctionType::get(double_type, args_d2, false));

        /// double fabs(double x);
        createCLibraryFunction(LibFunc_fabs,
                               FunctionType::get(double_type, args_d1, false));

        /// double acos(double x);
        createCLibraryFunction(LibFunc_acos,
                               FunctionType::get(double_type, args_d1, false));

        /// double asin(double x);
        createCLibraryFunction(LibFunc_asin,
                               FunctionType::get(double_type, args_d1, false));

        /// double atan(double x);
        createCLibraryFunction(LibFunc_atan,
                               FunctionType::get(double_type, args_d1, false));

        /// double ceil(double x);
        createCLibraryFunction(LibFunc_ceil,
                               FunctionType::get(double_type, args_d1, false));

        /// double cos(double x);
        createCLibraryFunction(LibFunc_cos,
                               FunctionType::get(double_type, args_d1, false));

        /// double cosh(double x);
        createCLibraryFunction(LibFunc_cosh,
                               FunctionType::get(double_type, args_d1, false));

        /// double exp(double x);
        createCLibraryFunction(LibFunc_exp,
                               FunctionType::get(double_type, args_d1, false));

        /// double floor(double x);
        createCLibraryFunction(LibFunc_floor,
                               FunctionType::get(double_type, args_d1, false));

        /// double log(double x);
        createCLibraryFunction(LibFunc_log,
                               FunctionType::get(double_type, args_d1, false));

        /// double log10(double x);
        createCLibraryFunction(LibFunc_log10,
                               FunctionType::get(double_type, args_d1, false));

        /// double sin(double x);
        createCLibraryFunction(LibFunc_sin,
                               FunctionType::get(double_type, args_d1, false));

        /// double sinh(double x);
        createCLibraryFunction(LibFunc_sinh,
                               FunctionType::get(double_type, args_d1, false));

        /// double tan(double x);
        createCLibraryFunction(LibFunc_tan,
                               FunctionType::get(double_type, args_d1, false));

        /// double tanh(double x);
        createCLibraryFunction(LibFunc_tanh,
                               FunctionType::get(double_type, args_d1, false));

        /// double fmod(double x, double y);
        createCLibraryFunction(LibFunc_fmod,
                               FunctionType::get(double_type, args_d2, false));
    }

    /**
    * getProcessTriple() - Return an appropriate target triple for generating
    * code to be loaded into the current process, e.g. when using the JIT.
    */
    std::string Jit::getProcessTriple() const {
        return llvm::sys::getProcessTriple();
    }

    std::string Jit::getDefaultTargetTriple() const {
        return llvm::sys::getDefaultTargetTriple();
    }


    const llvm::Target *Jit::getDefaultTargetMachine() const {
        std::string err;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget(getDefaultTargetTriple(), err);

        // Print an error and exit if we couldn't find the requested target.
        // This generally occurs if we've forgotten to initialise the
        // TargetRegistry or we have a bogus target triple.
        if (!target) {
            rrLogErr << err;
            llvm::errs() << err;
        }
        return target;
    }

    FnMap Jit::externalFunctionSignatures() const {
        llvm::Type *double_type = llvm::Type::getDoubleTy(*context);
        llvm::Type *int_type = llvm::Type::getInt32Ty(*context);
        llvm::Type *voidPtrType = llvm::Type::getInt8PtrTy(*context);

        llvm::Type *args_i1[] = {int_type};
        llvm::Type *args_d1[] = {double_type};
        llvm::Type *args_d2[] = {double_type, double_type};
        llvm::Type *args_void_double_quadruple[] = {voidPtrType, double_type, double_type, double_type, double_type};
        llvm::Type *args_void_double_triple[] = {voidPtrType, double_type, double_type, double_type};
        llvm::Type *args_void_double_double[] = {voidPtrType, double_type, double_type};
        llvm::Type *args_void_double[] = {voidPtrType, double_type};

        llvm::FunctionType *fnTy_double_argsd1 = llvm::FunctionType::get(double_type, args_d1, false);
        llvm::FunctionType *fnTy_double_argsd2 = llvm::FunctionType::get(double_type, args_d2, false);
        llvm::FunctionType *fnTy_int_argsi1 = llvm::FunctionType::get(int_type, args_i1, false);
        llvm::FunctionType *fnTy_double_void_d4 = llvm::FunctionType::get(double_type, args_void_double_quadruple,false);
        llvm::FunctionType *fnTy_double_void_d3 = llvm::FunctionType::get(double_type, args_void_double_triple, false);
        llvm::FunctionType *fnTy_double_void_d2 = llvm::FunctionType::get(double_type, args_void_double_double, false);
        llvm::FunctionType *fnTy_double_void_d1 = llvm::FunctionType::get(double_type, args_void_double, false);


        FnMap fnTyMap{
                {"arccot", {fnTy_double_argsd1, (void*)&sbmlsupport::arccot}},
                {"rr_arccot_negzero", {fnTy_double_argsd1, (void*)&sbmlsupport::rr_arccot_negzero}},
                {"arccoth", {fnTy_double_argsd1, (void*)&sbmlsupport::arccoth}},
                {"arccsc", {fnTy_double_argsd1, (void*)&sbmlsupport::arccsc}},
                {"arccsch", {fnTy_double_argsd1, (void*)&sbmlsupport::arccsch}},
                {"arcsec", {fnTy_double_argsd1, (void*)&sbmlsupport::arcsec}},
                {"arcsech", {fnTy_double_argsd1, (void*)&sbmlsupport::arcsech}},
                {"cot", {fnTy_double_argsd1, (void*)&sbmlsupport::cot}},
                {"coth", {fnTy_double_argsd1, (void*)&sbmlsupport::coth}},
                {"csc", {fnTy_double_argsd1, (void*)&sbmlsupport::csc}},
                {"csch", {fnTy_double_argsd1, (void*)&sbmlsupport::csch}},
                {"rr_factoriali", {fnTy_int_argsi1, (void*)&sbmlsupport::rr_factoriali}},
                {"rr_factoriald", {fnTy_double_argsd1, (void*)&sbmlsupport::rr_factoriald}},
                {"rr_logd", {fnTy_double_argsd2, (void*)&sbmlsupport::rr_logd}},
                {"rr_rootd", {fnTy_double_argsd2, (void*)&sbmlsupport::rr_rootd}},
                {"sec", {fnTy_double_argsd1, (void*)&sbmlsupport::sec}},
                {"sech", {fnTy_double_argsd1, (void*)&sbmlsupport::sech}},
                {"arccosh", {fnTy_double_argsd1, (void*)&sbmlsupport::arccosh}},
                {"arcsinh", {fnTy_double_argsd1, (void*)&sbmlsupport::arcsinh}},
                {"arctanh", {fnTy_double_argsd1, (void*)&sbmlsupport::arctanh}},
                {"quotient", {fnTy_double_argsd2, (void*)&sbmlsupport::quotient}},
                {"rr_max", {fnTy_double_argsd2, (void*)&sbmlsupport::rr_max}},
                {"rr_min", {fnTy_double_argsd2, (void*)&sbmlsupport::rr_min}},
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
                {"rr_distrib_bernoulli", {fnTy_double_void_d1, (void*)&distrib_bernoulli}},
                {"rr_distrib_cauchy_one", {fnTy_double_void_d1, (void*)&distrib_cauchy_one}},
                {"rr_distrib_laplace_one", {fnTy_double_void_d1, (void*)&distrib_laplace_one}},
                {"rr_distrib_exponential", {fnTy_double_void_d1, (void*)&distrib_exponential}},
                {"rr_distrib_uniform", {fnTy_double_void_d2, (void*)&distrib_uniform}},
                {"rr_distrib_normal", {fnTy_double_void_d2, (void*)&distrib_normal}},
                {"rr_distrib_binomial", {fnTy_double_void_d2, (void*)&distrib_binomial}},
                {"rr_distrib_cauchy", {fnTy_double_void_d2, (void*)&distrib_cauchy}},
                {"rr_distrib_chisquare", {fnTy_double_void_d1, (void*)&distrib_chisquare}},
                {"rr_distrib_gamma", {fnTy_double_void_d2, (void*)&distrib_gamma}},
                {"rr_distrib_laplace", {fnTy_double_void_d2, (void*)&distrib_laplace}},
                {"rr_distrib_lognormal", {fnTy_double_void_d2, (void*)&distrib_lognormal}},
                {"rr_distrib_poisson", {fnTy_double_void_d1, (void*)&distrib_poisson}},
                {"rr_distrib_rayleigh", {fnTy_double_void_d1, (void*)&distrib_rayleigh}},
                {"rr_distrib_chisquare_three", {fnTy_double_void_d3, (void*)&distrib_chisquare_three}},
                {"rr_distrib_exponential_three", {fnTy_double_void_d3, (void*)&distrib_exponential_three}},
                {"rr_distrib_poisson_three", {fnTy_double_void_d3, (void*)&distrib_poisson_three}},
                {"rr_distrib_rayleigh_three", {fnTy_double_void_d3, (void*)&distrib_rayleigh_three}},
                {"rr_distrib_normal_four", {fnTy_double_void_d4, (void*)&distrib_normal_four}},
                {"rr_distrib_binomial_four", {fnTy_double_void_d4, (void*)&distrib_binomial_four}},
                {"rr_distrib_cauchy_four", {fnTy_double_void_d4, (void*)&distrib_cauchy_four}},
                {"rr_distrib_gamma_four", {fnTy_double_void_d4, (void*)&distrib_gamma_four}},
                {"rr_distrib_laplace_four", {fnTy_double_void_d4, (void*)&distrib_laplace_four}},
                {"rr_distrib_lognormal_four", {fnTy_double_void_d4, (void*)&distrib_lognormal_four}},
#endif
        };

        return fnTyMap;
    }


}


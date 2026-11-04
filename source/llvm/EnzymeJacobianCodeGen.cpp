#include "EnzymeJacobianCodeGen.h"

#include "EvalRateRuleRatesCodeGen.h"
#include "EvalReactionRatesCodeGen.h"
#include "JacobianSparsity.h"
#include "LLVMModelDataSymbols.h"
#include "ModelGeneratorContext.h"
#include "rrRoadRunnerOptions.h"
#include "rrLogger.h"

#include <sbml/SBMLTypes.h>
#include <chrono>
#include <mutex>

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <vector>

#ifdef RR_USE_ENZYME
// registers the "enzyme" module pass with a pass builder, from the static Enzyme library
extern "C" void registerEnzyme(llvm::PassBuilder &PB);
#endif

namespace rrllvm {

    const char *EnzymeJacobianCodeGen::ReactionRatesTangentsName = "evalReactionRatesTangents";
    const char *EnzymeJacobianCodeGen::RateRuleRatesTangentsName = "evalRateRuleRatesTangents";

    namespace {
        /** Enzyme recognises its activity markers by the name of the global they reference. */
        llvm::GlobalVariable *marker(llvm::Module &module, const std::string &name) {
            if (auto *gv = module.getGlobalVariable(name, true)) {
                return gv;
            }
            auto *i32 = llvm::Type::getInt32Ty(module.getContext());
            return new llvm::GlobalVariable(module, i32, true, llvm::GlobalValue::InternalLinkage,
                                            llvm::ConstantInt::get(i32, 0), name);
        }
    }

    int EnzymeJacobianCodeGen::emitTangents(llvm::Module &module, const std::string &primal,
                                            const std::string &name, unsigned width, unsigned blockSize) {
        llvm::LLVMContext &context = module.getContext();
        llvm::Function *f = module.getFunction(primal);
        if (f == nullptr || f->empty()) {
            throw std::runtime_error("Enzyme: no generated function '" + primal + "' to differentiate");
        }

        auto *ptrTy = llvm::PointerType::getUnqual(context);
        auto *voidTy = llvm::Type::getVoidTy(context);
        auto *i64 = llvm::Type::getInt64Ty(context);

        auto *fnTy = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, false);
        auto *wrapper = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, name, module);
        llvm::Value *md = wrapper->getArg(0);
        llvm::Value *shadows = wrapper->getArg(1);
        md->setName("modelData");
        shadows->setName("shadows");

        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", wrapper));

        std::vector<llvm::Value *> shadowArgs;
        for (unsigned k = 0; k < width; ++k) {
            llvm::Value *slot = builder.CreateConstGEP1_64(ptrTy, shadows, k);
            shadowArgs.push_back(builder.CreateLoad(ptrTy, slot, "shadow" + std::to_string(k)));
        }

        // the derivative of the return value: a double per direction, or nothing for void
        llvm::Type *retTy = f->getReturnType();
        llvm::Type *diffRetTy = voidTy;
        if (!retTy->isVoidTy()) {
            diffRetTy = width == 1 ? retTy : llvm::ArrayType::get(retTy, width);
        }
        auto *enzymeTy = llvm::FunctionType::get(diffRetTy, {ptrTy}, true);
        llvm::FunctionCallee enzyme = module.getOrInsertFunction("__enzyme_fwddiff", enzymeTy);
        std::vector<llvm::Function *> blocks = splitByStores(f, blockSize);
        for (llvm::Function *block: blocks) {
            std::vector<llvm::Value *> args = {block, marker(module, "enzyme_width"),
                                               llvm::ConstantInt::get(i64, width), marker(module, "enzyme_dup"), md};
            args.insert(args.end(), shadowArgs.begin(), shadowArgs.end());
            builder.CreateCall(enzyme, args);
        }
        builder.CreateRetVoid();

        std::string err;
        llvm::raw_string_ostream os(err);
        if (llvm::verifyFunction(*wrapper, &os)) {
            throw std::runtime_error("Enzyme: invalid tangents function '" + name + "': " + os.str());
        }
        return static_cast<int>(blocks.size());
    }

    namespace {
        /**
         * Turn the declaration `name` into an internal definition built by body, which gets
         * an IRBuilder at the entry block and the arguments.
         */
        template<typename Body>
        void define(llvm::Module &module, const char *name, Body body) {
            llvm::Function *f = module.getFunction(name);
            if (f == nullptr || !f->empty()) {
                return;
            }
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
            f->addFnAttr(llvm::Attribute::AlwaysInline);
            llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(), "entry", f));
            std::vector<llvm::Value *> args;
            for (llvm::Argument &a: f->args()) {
                args.push_back(&a);
            }
            builder.CreateRet(body(builder, args));
        }

        llvm::Value *callLibm(llvm::Module &module, llvm::IRBuilder<> &b, const char *name, llvm::Value *x) {
            auto *d = b.getDoubleTy();
            llvm::FunctionCallee fn = module.getOrInsertFunction(name, llvm::FunctionType::get(d, {d}, false));
            return b.CreateCall(fn, {x});
        }
    }

    void EnzymeJacobianCodeGen::defineDifferentiableHelpers(llvm::Module &module) {
        using B = llvm::IRBuilder<>;
        using Args = std::vector<llvm::Value *>;
        // a > b ? a : b
        define(module, "rr_max", [](B &b, Args &a) {
            return b.CreateSelect(b.CreateFCmpOGT(a[0], a[1]), a[0], a[1]);
        });
        // a < b ? a : b
        define(module, "rr_min", [](B &b, Args &a) {
            return b.CreateSelect(b.CreateFCmpOLT(a[0], a[1]), a[0], a[1]);
        });
        // log(number) / log(base)
        define(module, "rr_logd", [&module](B &b, Args &a) {
            return b.CreateFDiv(callLibm(module, b, "log", a[1]), callLibm(module, b, "log", a[0]));
        });
        // a != 0 ? pow(b, 1/a) : 1
        define(module, "rr_rootd", [&module](B &b, Args &a) {
            auto *d = b.getDoubleTy();
            llvm::FunctionCallee pow = module.getOrInsertFunction("pow", llvm::FunctionType::get(d, {d, d}, false));
            llvm::Value *root = b.CreateCall(pow, {a[1], b.CreateFDiv(llvm::ConstantFP::get(d, 1.0), a[0])});
            return b.CreateSelect(b.CreateFCmpUNE(a[0], llvm::ConstantFP::get(d, 0.0)), root,
                                  llvm::ConstantFP::get(d, 1.0));
        });
        // a == 0 ? pi/2 : atan(1/a), and -pi/2 at zero for the negative-zero variant
        for (auto [name, atZero]: {std::pair<const char *, double>{"arccot", M_PI / 2.},
                                   std::pair<const char *, double>{"rr_arccot_negzero", -M_PI / 2.}}) {
            define(module, name, [&module, atZero = atZero](B &b, Args &a) {
                auto *d = b.getDoubleTy();
                llvm::Value *v = callLibm(module, b, "atan", b.CreateFDiv(llvm::ConstantFP::get(d, 1.0), a[0]));
                return b.CreateSelect(b.CreateFCmpOEQ(a[0], llvm::ConstantFP::get(d, 0.0)),
                                      llvm::ConstantFP::get(d, atZero), v);
            });
        }
    }

    namespace {
        /** Collects the error diagnostics of a context instead of terminating the process. */
        struct CollectingDiagnosticHandler : public llvm::DiagnosticHandler {
            std::vector<std::string> *errors;

            explicit CollectingDiagnosticHandler(std::vector<std::string> *errors) : errors(errors) {}

            bool handleDiagnostics(const llvm::DiagnosticInfo &info) override {
                if (info.getSeverity() == llvm::DS_Error) {
                    std::string msg;
                    llvm::raw_string_ostream os(msg);
                    llvm::DiagnosticPrinterRawOStream printer(os);
                    info.print(printer);
                    errors->push_back(os.str());
                }
                return true;
            }
        };
    }

#ifdef RR_USE_ENZYME
    namespace {
        /**
         * Sets Enzyme's command line options from RR_ENZYME_OPTIONS, a comma separated list of
         * name=value pairs (e.g. "enzyme-preopt=0"), once per process. For experiments only.
         */
        void applyEnzymeOptionsFromEnvironment() {
            static std::once_flag once;
            std::call_once(once, []() {
                const char *env = std::getenv("RR_ENZYME_OPTIONS");
                if (env == nullptr) {
                    return;
                }
                auto registered = llvm::cl::getRegisteredOptions();
                llvm::SmallVector<llvm::StringRef> items;
                llvm::StringRef(env).split(items, ',', -1, false);
                for (llvm::StringRef item: items) {
                    auto [name, value] = item.trim().split('=');
                    auto it = registered.find(name);
                    if (it == registered.end()) {
                        rrLogWarn << "RR_ENZYME_OPTIONS: unknown option '" << name.str() << "'";
                        continue;
                    }
                    it->second->reset();
                    if (it->second->addOccurrence(0, name, value)) {
                        rrLogWarn << "RR_ENZYME_OPTIONS: invalid value for '" << name.str() << "'";
                    } else {
                        rrLogInfo << "RR_ENZYME_OPTIONS: " << name.str() << " = " << value.str();
                    }
                }
            });
        }
    }
#endif

    void EnzymeJacobianCodeGen::runEnzymePass(llvm::Module &module) {
#ifndef RR_USE_ENZYME
        throw std::runtime_error("roadrunner was built without Enzyme (CMake option RR_USE_ENZYME)");
#else
        applyEnzymeOptionsFromEnvironment();
        llvm::LLVMContext &context = module.getContext();

        // Enzyme reports failures as error diagnostics, which terminate the process with the
        // default handler; collect them instead and restore the previous handler afterwards.
        std::vector<std::string> errors;
        std::unique_ptr<llvm::DiagnosticHandler> previous = context.getDiagnosticHandler();
        context.setDiagnosticHandler(std::make_unique<CollectingDiagnosticHandler>(&errors));

        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        llvm::PassBuilder pb;
        registerEnzyme(pb);
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;
        if (llvm::Error e = pb.parsePassPipeline(mpm, "enzyme")) {
            context.setDiagnosticHandler(std::move(previous));
            throw std::runtime_error("Enzyme: cannot build the pass pipeline: " + llvm::toString(std::move(e)));
        }
        mpm.run(module, mam);
        context.setDiagnosticHandler(std::move(previous));

        if (!errors.empty()) {
            std::string msg = "Enzyme failed on the generated model code:";
            for (const std::string &e: errors) {
                msg += "\n  " + e;
            }
            throw std::runtime_error(msg);
        }

        std::string err;
        llvm::raw_string_ostream os(err);
        if (llvm::verifyModule(module, &os)) {
            throw std::runtime_error("Enzyme produced an invalid module: " + os.str());
        }
#endif
    }


    namespace {
        /** Math library functions Enzyme differentiates. */
        const std::set<std::string> &differentiableLibm() {
            static const std::set<std::string> names = {
                    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "pow", "sqrt", "cbrt", "hypot",
                    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh",
                    "acosh", "atanh", "fabs", "floor", "ceil", "trunc", "round", "fmod", "erf", "erfc"};
            return names;
        }

        /**
         * Throw if a function reachable from the roots calls an external function Enzyme cannot
         * differentiate: a C++ helper without an IR body.
         */
        void screenCalls(llvm::Module &module, const std::vector<std::string> &roots) {
            std::vector<llvm::Function *> work;
            std::set<llvm::Function *> seen;
            for (const std::string &r: roots) {
                if (llvm::Function *f = module.getFunction(r)) {
                    work.push_back(f);
                    seen.insert(f);
                }
            }
            while (!work.empty()) {
                llvm::Function *f = work.back();
                work.pop_back();
                for (llvm::BasicBlock &bb: *f) {
                    for (llvm::Instruction &inst: bb) {
                        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
                        if (call == nullptr) {
                            continue;
                        }
                        llvm::Function *callee = call->getCalledFunction();
                        if (callee == nullptr) {
                            throw std::runtime_error("the generated code of '" + f->getName().str() +
                                                     "' makes an indirect call");
                        }
                        if (callee->isIntrinsic()) {
                            continue;
                        }
                        if (callee->isDeclaration()) {
                            if (!differentiableLibm().count(callee->getName().str())) {
                                throw std::runtime_error("the model calls '" + callee->getName().str() +
                                                         "', which has no derivative in Enzyme");
                            }
                            continue;
                        }
                        if (seen.insert(callee).second) {
                            work.push_back(callee);
                        }
                    }
                }
            }
        }

        /** Erase every function and global of the module that is not in the snapshots. */
        void eraseNewSymbols(llvm::Module &module, const std::set<llvm::Function *> &functions,
                             const std::set<llvm::GlobalVariable *> &globals) {
            std::vector<llvm::Function *> newFunctions;
            for (llvm::Function &f: module) {
                if (!functions.count(&f)) {
                    newFunctions.push_back(&f);
                }
            }
            for (llvm::Function *f: newFunctions) {
                f->dropAllReferences();
            }
            for (llvm::Function *f: newFunctions) {
                f->replaceAllUsesWith(llvm::PoisonValue::get(f->getType()));
                f->eraseFromParent();
            }
            std::vector<llvm::GlobalVariable *> newGlobals;
            for (llvm::GlobalVariable &g: module.globals()) {
                if (!globals.count(&g)) {
                    newGlobals.push_back(&g);
                }
            }
            for (llvm::GlobalVariable *g: newGlobals) {
                g->replaceAllUsesWith(llvm::PoisonValue::get(g->getType()));
                g->eraseFromParent();
            }
        }

        /** Why the model is outside what the Enzyme Jacobian covers, or an empty string. */
        std::string unsupported(const libsbml::Model &model, const LLVMModelDataSymbols &symbols) {
            for (unsigned i = 0; i < model.getNumReactions(); ++i) {
                const libsbml::Reaction *r = model.getReaction(i);
                for (const libsbml::ListOfSpeciesReferences *refs: {r->getListOfReactants(), r->getListOfProducts()}) {
                    for (unsigned k = 0; k < refs->size(); ++k) {
                        auto *ref = static_cast<const libsbml::SpeciesReference *>(refs->get(k));
                        bool ruled = ref->isSetId() && !ref->getId().empty() &&
                                     (symbols.hasAssignmentRule(ref->getId()) || symbols.hasRateRule(ref->getId()));
                        if (ruled || ref->isSetStoichiometryMath()) {
                            return "the stoichiometry of reaction '" + r->getId() +
                                   "' is set by a rule (volatile stoichiometry)";
                        }
                    }
                }
            }
            for (unsigned i = 0; i < model.getNumSpecies(); ++i) {
                const libsbml::Species *s = model.getSpecies(i);
                if (s->isSetConversionFactor()) {
                    return "species '" + s->getId() + "' has a conversion factor";
                }
            }
            return "";
        }
    }

    namespace {
        /** The helpers defineDifferentiableHelpers gives an IR body, all pure. */
        const std::set<std::string> &helpersWithBody() {
            static const std::set<std::string> names = {"rr_max", "rr_min", "rr_logd", "rr_rootd", "arccot",
                                                        "rr_arccot_negzero"};
            return names;
        }

        /** A call without side effects: a pure intrinsic, a math library function or a helper with an IR body. */
        bool isPureCall(const llvm::CallBase &call) {
            const llvm::Function *callee = call.getCalledFunction();
            if (callee == nullptr) {
                return false;
            }
            if (callee->isIntrinsic()) {
                return !call.mayHaveSideEffects();
            }
            const std::string name = callee->getName().str();
            return differentiableLibm().count(name) || helpersWithBody().count(name);
        }

        /** The struct field of the model data md that ptr points into, through GEPs, or -1. */
        int modelDataField(const llvm::Value *ptr, const llvm::Value *md) {
            auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
            while (gep != nullptr && gep->getPointerOperand() != md) {
                gep = llvm::dyn_cast<llvm::GEPOperator>(gep->getPointerOperand());
            }
            if (gep == nullptr || gep->getNumIndices() < 2) {
                return -1;
            }
            auto *first = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1));
            auto *field = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(2));
            return first != nullptr && first->isZero() && field != nullptr ? static_cast<int>(field->getZExtValue()) : -1;
        }

        /**
         * True if the load cannot read the reaction rates: it reads another field of the model
         * data, or through an alias pointer of the model data other than ReactionRatesAlias.
         */
        bool loadAvoidsReactionRates(const llvm::LoadInst &load, const llvm::Value *md) {
            const llvm::Value *ptr = load.getPointerOperand();
            int field = modelDataField(ptr, md);
            if (field >= 0) {
                return field != ReactionRates;
            }
            const llvm::Value *base = ptr;
            while (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(base)) {
                base = gep->getPointerOperand();
            }
            auto *aliasLoad = llvm::dyn_cast<llvm::LoadInst>(base);
            if (aliasLoad == nullptr) {
                return false;
            }
            int aliasField = modelDataField(aliasLoad->getPointerOperand(), md);
            return aliasField >= 0 && aliasField != ReactionRatesAlias;
        }

        /** Erase every instruction whose result is unused and that has no effect, pure calls included. */
        void eraseDeadCode(llvm::Function &f) {
            auto removable = [](llvm::Instruction &inst) {
                if (!inst.use_empty() || inst.isTerminator()) {
                    return false;
                }
                auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
                return llvm::isInstructionTriviallyDead(&inst) || (call != nullptr && isPureCall(*call));
            };
            std::vector<llvm::Instruction *> work;
            for (llvm::BasicBlock &bb: f) {
                for (llvm::Instruction &inst: bb) {
                    if (removable(inst)) {
                        work.push_back(&inst);
                    }
                }
            }
            while (!work.empty()) {
                llvm::Instruction *inst = work.back();
                work.pop_back();
                std::set<llvm::Instruction *> operands;
                for (llvm::Use &u: inst->operands()) {
                    if (auto *op = llvm::dyn_cast<llvm::Instruction>(u.get())) {
                        operands.insert(op);
                    }
                }
                inst->eraseFromParent();
                for (llvm::Instruction *op: operands) {
                    if (removable(*op)) {
                        work.push_back(op);
                    }
                }
            }
        }
    }

    std::vector<llvm::Function *> EnzymeJacobianCodeGen::splitByStores(llvm::Function *f, unsigned blockSize) {
        if (f->arg_empty()) {
            return {f};
        }
        const llvm::Value *md = f->getArg(0);
        std::vector<llvm::StoreInst *> stores;
        for (llvm::BasicBlock &bb: *f) {
            for (llvm::Instruction &inst: bb) {
                if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
                    if (modelDataField(store->getPointerOperand(), md) != ReactionRates) {
                        return {f};
                    }
                    stores.push_back(store);
                } else if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
                    if (!loadAvoidsReactionRates(*load, md)) {
                        return {f};
                    }
                } else if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                    if (!isPureCall(*call)) {
                        return {f};
                    }
                } else if (inst.mayWriteToMemory()) {
                    return {f};
                }
            }
        }
        if (blockSize == 0 || stores.size() <= blockSize) {
            return {f};
        }
        std::vector<llvm::Function *> blocks;
        for (size_t begin = 0; begin < stores.size(); begin += blockSize) {
            size_t end = std::min(stores.size(), begin + blockSize);
            llvm::ValueToValueMapTy map;
            llvm::Function *block = llvm::CloneFunction(f, map);
            block->setName(f->getName() + ".block" + std::to_string(blocks.size()));
            block->setLinkage(llvm::GlobalValue::InternalLinkage);
            for (size_t i = 0; i < stores.size(); ++i) {
                if (i < begin || i >= end) {
                    llvm::cast<llvm::StoreInst>(map[stores[i]])->eraseFromParent();
                }
            }
            eraseDeadCode(*block);
            blocks.push_back(block);
        }
        return blocks;
    }

    void EnzymeJacobianCodeGen::prepare(const ModelGeneratorContext &context, unsigned options,
                                        EnzymeJacobianResources &res) {
        res = EnzymeJacobianResources();
#ifndef RR_USE_ENZYME
        res.unavailableReason = "roadrunner was built without Enzyme (CMake option RR_USE_ENZYME)";
#else
        if (options & rr::LoadSBMLOptions::CONSERVED_MOIETIES) {
            res.unavailableReason = "conserved moiety analysis is switched on";
            return;
        }
        auto start = std::chrono::steady_clock::now();
        const libsbml::Model &model = *context.getModel();
        const LLVMModelDataSymbols &symbols = context.getModelDataSymbols();
        std::string reason = unsupported(model, symbols);
        if (!reason.empty()) {
            res.unavailableReason = reason;
            return;
        }
        JacobianSparsity sparsity(model, symbols);
        if (model.isSetConversionFactor() && sparsity.dependsOnState(model.getConversionFactor())) {
            res.unavailableReason = "the conversion factor '" + model.getConversionFactor() + "' depends on the state";
            return;
        }

        res.reactionRates = sparsity.reactionRates();
        res.rateRules = sparsity.rateRules();
        res.jacobian = sparsity.jacobian();
        res.colors = sparsity.colors();
        res.numColors = sparsity.numColors();
        // a larger width needs fewer passes per Jacobian and compiles much slower
        int maxWidth = DefaultWidth;
        if (const char *env = std::getenv("RR_ENZYME_MAX_WIDTH")) {
            maxWidth = std::max(1, std::atoi(env));
        }
        res.width = std::min(res.numColors, maxWidth);
        if (res.width == 0) {
            res.unavailableReason.clear();  // no states: an empty Jacobian, nothing to generate
            return;
        }

        llvm::Module &module = *context.getJitNonOwning()->getModuleNonOwning();
        std::set<llvm::Function *> functions;
        for (llvm::Function &f: module) {
            functions.insert(&f);
        }
        std::set<llvm::GlobalVariable *> globals;
        for (llvm::GlobalVariable &g: module.globals()) {
            globals.insert(&g);
        }
        try {
            defineDifferentiableHelpers(module);
            screenCalls(module, {EvalReactionRatesCodeGen::FunctionName, EvalRateRuleRatesCodeGen::FunctionName});
            int blockSize = DefaultBlockSize;
            if (const char *env = std::getenv("RR_ENZYME_BLOCK_SIZE")) {
                blockSize = std::max(1, std::atoi(env));
            }
            res.reactionRateBlocks = emitTangents(module, EvalReactionRatesCodeGen::FunctionName,
                                                  ReactionRatesTangentsName, res.width, blockSize);
            res.rateRuleBlocks = emitTangents(module, EvalRateRuleRatesCodeGen::FunctionName,
                                              RateRuleRatesTangentsName, res.width, blockSize);
            if (const char *dump = std::getenv("RR_ENZYME_DUMP_IR")) {
                // for experiments: the module as Enzyme gets it
                std::error_code ec;
                llvm::raw_fd_ostream out(dump, ec);
                if (!ec) {
                    module.print(out, nullptr);
                }
            }
            auto enzymeStart = std::chrono::steady_clock::now();
            runEnzymePass(module);
            auto enzymeEnd = std::chrono::steady_clock::now();
            // the blocks are only the input of Enzyme; nothing calls them, so do not compile them
            std::vector<llvm::Function *> unused;
            for (llvm::Function &g: module) {
                if (!functions.count(&g) && g.hasInternalLinkage() && g.use_empty()) {
                    unused.push_back(&g);
                }
            }
            for (llvm::Function *g: unused) {
                g->eraseFromParent();
            }
            rrLog(rr::Logger::LOG_DEBUG) << "Enzyme Jacobian: removed " << unused.size() << " unused functions";
            rrLog(rr::Logger::LOG_INFORMATION) << "Enzyme Jacobian: " << res.reactionRateBlocks
                    << " reaction rate blocks, sparsity and IR setup "
                    << std::chrono::duration<double>(enzymeStart - start).count() << " s, Enzyme pass "
                    << std::chrono::duration<double>(enzymeEnd - enzymeStart).count() << " s";
            res.unavailableReason.clear();
        } catch (const std::exception &e) {
            eraseNewSymbols(module, functions, globals);
            EnzymeJacobianResources failed;
            failed.unavailableReason = e.what();
            res = failed;
        }
#endif
    }

    void EnzymeJacobianCodeGen::lookup(const ModelGeneratorContext &context, EnzymeJacobianResources &res) {
        if (!res.available() || res.width == 0) {
            return;
        }
        Jit *jit = context.getJitNonOwning();
        res.reactionRatesTangents = reinterpret_cast<EnzymeTangents_FunctionPtr>(
                jit->lookupFunctionAddress(ReactionRatesTangentsName));
        res.rateRuleRatesTangents = reinterpret_cast<EnzymeTangents_FunctionPtr>(
                jit->lookupFunctionAddress(RateRuleRatesTangentsName));
    }

}

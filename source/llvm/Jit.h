//
// Created by Ciaran on 25/10/2021.
//

#ifndef ROADRUNNER_JIT_H
#define ROADRUNNER_JIT_H

#define NOMINMAX

#include <rrSparse.h>
#include "LLVMException.h"
#include "rrSparse.h"
#include <unordered_map>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include <llvm/Analysis/TargetLibraryInfo.h>
#include "llvm/IR/IRBuilder.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/IR/Mangler.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

namespace rr {
    class ExecutableModel;

}

namespace rrllvm {
    class Random;

    class ModelResources;

    class SBMLModelObjectCache;

    template<typename FunctionPtrType>
    class CodeGenBase;


    /**
     * @brief define some function pointer signatures
     */

    using FnPtr_d1 = double (*)(double x);
    using FnPtr_i1 = int (*)(int x);
    using FnPtr_d2 = double (*)(double x, double y);

    // these are taken from C library by createCLibraryFunctions
    using fabsFnTy = FnPtr_d1;
    using acosFnTy = FnPtr_d1;
    using asinFnTy = FnPtr_d1;
    using atanFnTy = FnPtr_d1;
    using ceilFnTy = FnPtr_d1;
    using cosFnTy = FnPtr_d1;
    using coshFnTy = FnPtr_d1;
    using expFnTy = FnPtr_d1;
    using floorFnTy = FnPtr_d1;
    using logFnTy = FnPtr_d1;
    using log10FnTy = FnPtr_d1;
    using sinFnTy = FnPtr_d1;
    using sinhFnTy = FnPtr_d1;
    using tanFnTy = FnPtr_d1;
    using tanhFnTy = FnPtr_d1;
    using fmodFnTy = FnPtr_d2;
    using powFnTy = FnPtr_d2;

    // these are defined locally by roadrunner in SBMLSupportFunctions
    using arccotFnTy = FnPtr_d1;
    using rr_arccot_negzeroFnTy = FnPtr_d1;
    using arccothFnTy = FnPtr_d1;
    using arccscFnTy = FnPtr_d1;
    using arccschFnTy = FnPtr_d1;
    using arcsecFnTy = FnPtr_d1;
    using arcsechFnTy = FnPtr_d1;
    using cotFnTy = FnPtr_d1;
    using cothFnTy = FnPtr_d1;
    using cscFnTy = FnPtr_d1;
    using cschFnTy = FnPtr_d1;
    using secFnTy = FnPtr_d1;
    using sechFnTy = FnPtr_d1;
    using arccoshFnTy = FnPtr_d1;
    using arcsinhFnTy = FnPtr_d1;
    using arctanhFnTy = FnPtr_d1;
    using rr_factorialdFnTy = FnPtr_d1;
    using rr_factorialiFnTy = FnPtr_i1;
    using rr_logdFnTy = FnPtr_d2;
    using rr_rootdFnTy = FnPtr_d2;
    using quotientFnTy = FnPtr_d2;
    using rr_maxFnTy = FnPtr_d2;
    using rr_minFnTy = FnPtr_d2;

    // for a sparse matrix used in llvm world
    using csr_matrix_set_nz_FnTy = rr::csr_matrix *(*)(int, int, double);
    using csr_matrix_get_nz_FnTy = rr::csr_matrix *(*)(int, int);

    // function signatures for distrib
    using DistribFnTy_d1 = double (*)(Random *, double);
    using DistribFnTy_d2 = double (*)(Random *, double, double);
    using DistribFnTy_d3 = double (*)(Random *, double, double, double);
    using DistribFnTy_d4 = double (*)(Random *, double, double, double, double);

    using distrib_bernoulli_FnTy = DistribFnTy_d1;
    using distrib_cauchy_one_FnTy = DistribFnTy_d1;
    using distrib_chisquare_FnTy = DistribFnTy_d1;
    using distrib_exponential_FnTy = DistribFnTy_d1;
    using distrib_laplace_one_FnTy = DistribFnTy_d1;
    using distrib_poisson_FnTy = DistribFnTy_d1;
    using distrib_rayleigh_FnTy = DistribFnTy_d1;
    using distrib_uniform_FnTy = DistribFnTy_d2;
    using distrib_normal_FnTy = DistribFnTy_d2;
    using distrib_binomial_FnTy = DistribFnTy_d2;
    using distrib_cauchy_FnTy = DistribFnTy_d2;
    using distrib_gamma_FnTy = DistribFnTy_d2;
    using distrib_laplace_FnTy = DistribFnTy_d2;
    using distrib_lognormal_FnTy = DistribFnTy_d2;
    using distrib_chisquare_three_FnTy = DistribFnTy_d3;
    using distrib_exponential_three_FnTy = DistribFnTy_d3;
    using distrib_poisson_three_FnTy = DistribFnTy_d3;
    using distrib_rayleigh_three_FnTy = DistribFnTy_d3;
    using distrib_normal_four_FnTy = DistribFnTy_d4;
    using distrib_binomial_four_FnTy = DistribFnTy_d4;
    using distrib_cauchy_four_FnTy = DistribFnTy_d4;
    using distrib_gamma_four_FnTy = DistribFnTy_d4;
    using distrib_laplace_four_FnTy = DistribFnTy_d4;
    using distrib_lognormal_four_FnTy = DistribFnTy_d4;

    using rrOwningBinary = llvm::object::OwningBinary<llvm::object::ObjectFile>;

    /**
     * {FunctionName : {functionType, (void*)FnAddress}}
     */
    using FnMap = std::unordered_map<std::string, std::pair<llvm::FunctionType *, void *>>;

    /**
     * @brief superclass of all Jit types. Builds the machinery necessary
     * to compile a sbml model on the fly to machine code.
     */
    class Jit {
    public:

        /**
         * @brief instantiate a Jit object with some options.
         * @param options LoadSBMLOptions::modelGeneratorOpt. The options are
         * stored by bit masks and can be manipulated by either modifying
         * LoadSBMLOptions or Config.
         */
        explicit Jit(std::uint32_t options);

        /**
         * @brief default constructor.
         * @details delegates to Jit(std::uint32_t options). The options
         * argument is the default constructed from LoadSBMLOptions.modelGeneratorOpt.
         * Note, that LoadSBMLOptions is influenced by the global Config.
         */
        Jit();

        /**
         * @brief adds functions that are declared and defined in C++ to the jit engine.
         * @details unlike most functions that are created directly in the llvm IR language
         * these functions are just C++ i.e. tan, arccsh, quotient. In MCJit (llvm 6) these
         * were "mapped" to sbml support functions in addGlobalMappings.
         */
        virtual void mapFunctionsToJitSymbols() = 0;

        /**
         * @brief Add support for libsbml distrib functions.
         * @details similar to Jit::mapFunctionsToJitSymbols, these are declared and defined
         * in C++. Support for distrib in libsbml must be present for these functions
         * to be jitted, but if distrib is not available there are no adverse consequences
         * other than no being able to simulate models that depend on libsbml distrib.
         */
//        virtual void mapDistribFunctionsToJitSymbols() = 0;

        /**
         * @brief defaulted virtual destructor
         */
        virtual ~Jit() = default;

        /**
         * @brief locate a function address based on its name. The returned
         * function address is a 64 bit int that needs casting to the function
         * pointer type.
         * @details throws an error if an address for function called @param name
         * is not found.
         * @example
         *   // given a function, for example:
         *   int fib(int x){
         *      if (x < 2)
         *          return x;
         *      return fib(x-1) + fib(x-2);
         *   }
         *
         *   // and assuming the function was jitt'ed and added to the module under the
         *   // symbol "fib"
         *   @code
         *   using fibonacciFnPtr = int (*)(int);
         *   fibonacciFnPtr fib = (fibonacciFnPtr)lookupFunctionAddress("fib");
         *   int fibOf4 = fib(4); // 3
         *   @endcode
         */
        virtual std::uint64_t lookupFunctionAddress(const std::string &name) = 0;

        /**
         * @brief add an in-memory representation of an object file to the current jit module.
         * @details the rrOwningBinary is a typedef'd
         * llvm::object::OwningBinary<llvm::object::ObjectFile>
         */
        virtual void addObjectFile(rrOwningBinary owningObject) = 0;

        /**
         * @brief add an in-memory representation of an object file to the current jit module.
         */
        virtual void addObjectFile(std::unique_ptr<llvm::object::ObjectFile> objectFile) = 0;

        /**
         * @brief add an in-memory representation of an object file to the current jit module.
         */
        virtual void addObjectFile(std::unique_ptr<llvm::MemoryBuffer> obj) = 0;

        /**
         * @brief add a module @param M directly to the jit engine.
         */
        virtual void addModule(llvm::Module *M) = 0;

        /**
         * @brief add a module @param M and Context @param ctx to the current jit engine.
         * @details the parameters M and cts are unique pointers and therefore must be moved, not
         * copied into the module. An llvm ThreadSafeModule is created internalls, which steals the
         * references. For this reason, any code to be added to the module must happen before the
         * call to addModule.
         */
        virtual void addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx) = 0;

        /**
         * @brief add the module and context which are member variables of the Jit instance to
         * the current Jit engine.
         * @details similar to addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx)
         * the references are stolen by the call to construct a ThreadSafeModule which then takes ownership of the
         * module and context.
         */
        virtual void addModule() = 0;

        /**
         * @brief get the DataLayout currently in use in the Jit
         */
        virtual const llvm::DataLayout &getDataLayout() const = 0;

        /**
         * @brief lookup the sbml with the md5 @param sbmlMD5 in the compiled object cache.
         * If it is there, return a memory buffer containing it (which you can turn into an object file).
         * Returns empty (null) unique_ptr if not found.
         * @note at present this only works with LLJit.
         */
        virtual std::unique_ptr<llvm::MemoryBuffer> getCompiledModelFromCache(const std::string &sbmlMD5) = 0;

        /**
         * @brief *Moves* objects over to ModelResources ptr
         */
        virtual void transferObjectsToResources(std::shared_ptr<rrllvm::ModelResources> modelResources);

        /**
         * @brief Map the Jit'd functions that collectively represent a
         * roadrunner compiled sbml model to symbols in the ModelResources.
         * @param modelResources the ModelResources to which to map.
         * @param options bitfield options to use.
         * @details this must be called after a the main roadrunner module
         * has been added to the Jit -- otherwise the lookup will fail.
         */
        virtual void mapLLVMGeneratedFunctionsToSymbols(ModelResources *modelResources, std::uint32_t options);

        /**
         * @brief returns a non owning pointer to the llvm::Module instance
         */
        virtual llvm::Module *getModuleNonOwning();

        /**
         * @brief returns a non owning pointer to the llvm::LLVMContext instance
         */
        virtual llvm::LLVMContext *getContextNonOwning();

        /**
         * @brief returns a non owning pointer to the llvm::LLVMContext instance
         */
        virtual llvm::IRBuilder<> *getBuilderNonOwning();

        /**
         * @brief Write the Jit::module in its current state
         * to string as LLVM IR.
         */
        virtual std::string emitToString();

        /**
         * @brief get the stream that stores a compiled module as binary.
         */
        llvm::raw_svector_ostream &getCompiledModuleStream();

        std::string getDefaultTargetTriple() const;

        void setModuleIdentifier(const std::string &id);

        virtual std::string mangleName(const std::string &unmangledName) const;

        /**
         * @brief get a binary string representation of the current
         * module.
         * @details This must be called after the module is added to the
         * jit engine. Throws if string is empty.
         *
         * This interface is moderate abuse of interface design -- the parameter
         * sbmlMD5 is only required for the LLJit subclass, as it is used to query the
         * object cache created by llvm. This is of course bad design
         * but we are boxed by not being able to find a way to enable the same cache
         * system with the old MCJit class.
         *
         * Regardless of the bad design, it works, so lets not spend any more time on it.
         */
        virtual std::string getModuleAsString(std::string sbmlMD5) = 0;

        /**
         * @brief Return MCJit's compiled binary stream as a string.
         * @details Converts the binary stream to a string.  Useful for saving and
         * loading state for MCJit objects (LLJit doesn't use it).
         */
        virtual std::string getModuleBinaryStreamAsString();

        /**
         * @brief Reset MCJit's compiled binary stream with the given string.
         * @details MCJit is the only Jit that uses this. (It's bad interface design
         * but works at least.). LLJit uses a caching mechanism which allows us
         * to retrieve object files directly, foregoing the need for this function.
         */
        virtual void resetModuleBinaryStream(std::string cmbs);

        /**
         * @brief MCJit compiles the generated LLVM IR to this binary stream
         * which is then used both for adding to the Jit as a module and for
         * saveState.
         * @details MCJit is the only Jit that uses this. (It's bad interface design
         * but works at least.). LLJit uses a caching mechanism which allows us
         * to retrieve object files directly, foregoing the need for this variable.
         */
        std::unique_ptr<llvm::raw_svector_ostream> compiledModuleBinaryStream;

        /**
         * The buffer used by compiledModuleBinaryStream
         */
        llvm::SmallVector<char, 10> moduleBuffer;
    protected:

        /**
         *
         */
        std::string getProcessTriple() const;

        /**
         * @brief use llvm calls to work out which TargetMachine
         * is currently being used.
         */
        const llvm::Target *getDefaultTargetMachine() const;

        /**
         * @brief returns a mapping between function names and function signatures
         */
        FnMap externalFunctionSignatures() const;


        std::unique_ptr<llvm::LLVMContext> context;
        std::unique_ptr<llvm::Module> module;
        llvm::Module *moduleNonOwning = nullptr;
        std::unique_ptr<llvm::IRBuilder<>> builder;
        std::uint32_t options;


    private:


        /**
         * @brief Add a function from the standard C library to the IR Module.
         * An example declaration is:
         * @code
         *   declare double pow(double, double)
         * @endcode
         * @details the declaration is resolved with the standard C
         * library.
         * @code
                using powFn = double (*)(double x, double y);
                powFn pow = (powFn) executionEngine->getPointerToNamedFunction("pow");
                std::cout << pow(4, 2) << std::endl; // outputs 16
         * @endcode
         */
        void createCLibraryFunction(llvm::LibFunc funcId, llvm::FunctionType *funcType);

        /**
         * @brief Pull a collection of functions from the standard C library into the developing LLVM IR Module
         * @details The interface for pulling in the C library functions (seems to) work for both llvm-6
         * and llvm-13. Therefore it is implemented in the superclass of all Jit's.
         * The following functions are declared:
         *    - pow, fabs, acos, asin, atan, ceil, cos, cosh, exp,
         *      floor, log, log10, sin, sinh, tan, tanh, fmod
         * @see Jit::mapFunctionsToJitSymbols
         */
        void createCLibraryFunctions();


    };
}

#endif //ROADRUNNER_JIT_H

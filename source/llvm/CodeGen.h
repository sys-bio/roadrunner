/*
 * CodeGen.h
 *
 *  Created on: Jul 2, 2013
 *      Author: andy
 */

#ifndef CodeGenH
#define CodeGenH

#include "LLVMIncludes.h"
#include <stack>
#include <string>

namespace rrllvm
{
    /** @class CodeGen
    * Base class for code generation.
    */
    class CodeGen
    {
    public:
        virtual llvm::Value* codeGen() = 0;
    protected:
        ~CodeGen() = default;;
    };

    /**
     * special name for the time symbol
     */
#define SBML_TIME_SYMBOL "\\time"

     /** @class LoadSymbolResolver
     * LLVM load instruction class
     */
    class LoadSymbolResolver
    {
    public:
        /**
         * generate an LLVM load instruction.
         *
         * @param symbol: the symbol name to resolve
         * @param args: function argument list
         */
        virtual llvm::Value* loadSymbolValue(
            const std::string& symbol,
            const llvm::ArrayRef<llvm::Value*>& args = llvm::ArrayRef<llvm::Value*>()) = 0;

        virtual void recursiveSymbolPush(const std::string& symbol) = 0;

        virtual void recursiveSymbolPop() = 0;

        virtual bool isLocalParameter(const std::string& symbol) = 0;

        /**
         * nested conditionals (or functions?) can push a local cache block, where
         * symbols would be chached. These need to be popped as these symbols are
         * not valid outside of the local conditional or scope block.
         */
        virtual size_t pushCacheBlock() { return 0; }

        /**
         * Pop a scoped cache block, this clears these values, and any subsequent reads
         * re-evaluate the requested symbol.
         *
         * Will throw an exception if an empty stack pop is attempted.
         */
        virtual size_t popCacheBlock() { return 0; }

    protected:

        virtual ~LoadSymbolResolver() {};
    };

    /** @class StoreSymbolResolver
    * LLVM store symbol class
    */
    class StoreSymbolResolver
    {
    public:
        virtual llvm::Value* storeSymbolValue(const std::string& symbol,
            llvm::Value* value) = 0;

    protected:
        ~StoreSymbolResolver() {};
    };



} /* namespace rr */
#endif /* LLVMCodeGenH */

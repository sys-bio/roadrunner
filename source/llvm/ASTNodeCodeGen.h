/*
 * ASTNodeCodeGen.h
 *
 * Created on: Jun 16, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#ifndef ASTNodeCodeGenH
#define ASTNodeCodeGenH

#include "LLVMIncludes.h"
#include "CodeGen.h"
#include "ModelGeneratorContext.h"
#include <thread>

static std::mutex ASTNodeMtx;

namespace libsbml
{
class ASTNode;
}

namespace rrllvm
{
/** @class ASTNodeCodeGen
 * All of the LLVM IR generation is handled here.
 */
class ASTNodeCodeGen
{
public:
    ASTNodeCodeGen(llvm::IRBuilder<> &builder,
            LoadSymbolResolver &resolver,
            const ModelGeneratorContext& ctx,
            llvm::Value *modelData);
    ~ASTNodeCodeGen();

    llvm::Value* codeGenDouble(const libsbml::ASTNode* ast);
    llvm::Value* codeGenBoolean(const libsbml::ASTNode* ast);
private:
    llvm::Value* codeGen(const libsbml::ASTNode* ast);

    /**
     * sbml ASTNode does not contain as assigment '=' type, assigment
     * is handled by other sbml elements such as initialAssigment.
     */
    llvm::Value *binaryExprCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *notImplemented(const libsbml::ASTNode *ast);

    llvm::Value *delayExprCodeGen(const libsbml::ASTNode *ast);

    llvm::Value* rateOfCodeGen(const libsbml::ASTNode* ast);

    llvm::Value *nameExprCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *realExprCodeGen(const libsbml::ASTNode *ast);

    /**
     * Support new AST nodes for distributions
     */
    llvm::Value *distribCodeGen(const libsbml::ASTNode *ast);


    /**
     * for now, just convert to double,
     *
     * TODO: is this right???
     */
    llvm::Value *integerCodeGen(const libsbml::ASTNode *ast);

    /**
     * would have made the most sense using templated pointer to a
     * IRBuilder member function, but many compilers seem to have trouble
     * with such templates.
     */
    llvm::Value *applyArithmeticCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *minmaxCodeGen(const libsbml::ASTNode *ast);

	// AHu: As of June'18, there is no difference between the two relation CodeGen
	// functions, so I am refactoring this function to just take two arguments,
	// and using it to be the only relational switch statement
    llvm::Value *applyBinaryRelationalCodeGen(const libsbml::ASTNode *ast,
		llvm::Value* left, llvm::Value* right);

    // JKM: NOTE: Not SBML-compliant, needed for idiosyncrasies in some legacy JDesigner models
    llvm::Value *applyScalarRelationalCodeGen(const libsbml::ASTNode *ast);

	// Ahu: I made this just so that later we can have an interface for non-scalar relations
	llvm::Value *applyRelationalCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *applyLogicalCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *functionCallCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *intrinsicCallCodeGen(const libsbml::ASTNode *ast);

    llvm::Value *piecewiseCodeGen(const libsbml::ASTNode *ast);

    /**
     * coerces a value to a boolean single bit.
     *
     * If value is already a boolean, it is unchanged.
     */
    llvm::Value *toBoolean(llvm::Value *value);

    /**
     * coerces a value to a double
     *
     * If value is already a double, it is unchanged.
     */
    llvm::Value *toDouble(llvm::Value* value);


    llvm::IRBuilder<> &builder;
    LoadSymbolResolver &resolver;
    const ModelGeneratorContext& ctx;
    llvm::Value *modelData;

    /**
     * get the module, only valid whilst a BasicBlock is being filled.
     *
     * @param fname: name of the calling function, used for generating exception
     * on failure.
     *
     * @returns a module on success, throws exeption on failure.
     */
    llvm::Module *getModule();

    bool scalar_mode_;

    friend class ASTNodeCodeGenScalarTicket;
};

std::string to_string(const libsbml::ASTNode *ast);

/** @class ASTNodeCodeGenScalarTicket
 *  Used in LoadSymbolResolverBase::loadReactionRate when resolving a kinetic law.
 */

class ASTNodeCodeGenScalarTicket {
    public:
        ASTNodeCodeGenScalarTicket(ASTNodeCodeGen& gen, bool val, std::string n = "");

        ~ASTNodeCodeGenScalarTicket();

    private:
        ASTNodeCodeGen& z_;
        bool v_;
        std::string n_;
};

} /* namespace rr */

#endif /* ASTNodeCodeGenH */

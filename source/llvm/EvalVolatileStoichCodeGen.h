/*
 * EvalVolatileStoichCodeGen.h
 *
 *  Created on: Aug 25, 2013
 *      Author: andy
 */

#ifndef EVALVOLATILESTOICHCODEGEN_H_
#define EVALVOLATILESTOICHCODEGEN_H_

#include "CodeGenBase.h"
#include "ModelGeneratorContext.h"
#include "SymbolForest.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include <sbml/Model.h>



namespace rrllvm
{

    typedef void (*EvalVolatileStoichCodeGen_FunctionPtr)(LLVMModelData*);

    /** @class EvalVolatileStoichCodeGen
    * Evaluate 'volatile stoichiometries', or, stoichiometries that can change over the course of a simulation.
    */
    class EvalVolatileStoichCodeGen :
        public CodeGenBase<EvalVolatileStoichCodeGen_FunctionPtr>
    {
    public:
        EvalVolatileStoichCodeGen(const ModelGeneratorContext& mgc);
        virtual ~EvalVolatileStoichCodeGen();

        llvm::Value* codeGen();

        static const char* FunctionName;

    private:
        /**
         * determine if the given reference dpends on any non-constant elements.
         *
         * only support non-constant species reference on l3v1 docs or above,
         * currently we can't determine if earlier version species references
         * are constant or not.
         */
        bool isConstantSpeciesReference(const
            libsbml::SimpleSpeciesReference* ref) const;

        /**
         * go through the AST tree and see if any names reference non-constant
         * document elements.
         */
        bool isConstantASTNode(const libsbml::ASTNode* ast) const;

        /**
         * update a single MultiSpeciesReference-typed named reference's
         * contribution to its shared stoichiometry-matrix cell by its own
         * delta (rawValue vs. its last known value in
         * multiSpeciesReferencesAlias), rather than overwriting the whole
         * cell -- overwriting would discard the other reference(s) sharing
         * that cell. Also updates multiSpeciesReferencesAlias[slot] to
         * rawValue.
         *
         * @param rawValue the reference's own value, unsigned (no sign
         * correction for reactant/product), as computed from its rate
         * rule / assignment rule / StoichiometryMath.
         */
        void codeGenMultiSpeciesReferenceUpdate(ModelDataIRBuilder& mdbuilder,
            const LLVMModelDataSymbols::SpeciesReferenceInfo& info,
            const std::string& id, llvm::Value* rawValue, bool isReactant);
    };

} /* namespace rrllvm */
#endif /* EVALVOLATILESTOICHCODEGEN_H_ */

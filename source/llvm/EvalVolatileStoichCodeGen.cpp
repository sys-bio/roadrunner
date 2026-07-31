/*
 * EvalVolatileStoichCodeGen.cpp
 *
 *  Created on: Aug 25, 2013
 *      Author: andy
 */
#pragma hdrstop

#include "EvalVolatileStoichCodeGen.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "ASTNodeFactory.h"
#include "ModelDataSymbolResolver.h"
#include "rrLogger.h"

#include <vector>
#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <Poco/Logger.h>

namespace rrllvm {
    using namespace rr;
    using namespace llvm;
    using namespace libsbml;


    const char *EvalVolatileStoichCodeGen::FunctionName = "evalVolatileStoich";

    EvalVolatileStoichCodeGen::EvalVolatileStoichCodeGen(
            const ModelGeneratorContext &mgc) :
            CodeGenBase<EvalVolatileStoichCodeGen_FunctionPtr>(mgc) {
    }

    EvalVolatileStoichCodeGen::~EvalVolatileStoichCodeGen() {
    }

    Value *EvalVolatileStoichCodeGen::codeGen() {
        Value *modelData = 0;

        codeGenVoidModelDataHeader(FunctionName, modelData);

        ModelDataLoadSymbolResolver resolver(modelData, modelGenContext);
        ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);

        LLVMModelDataSymbols* dataSymbolsPtr = const_cast<LLVMModelDataSymbols*>(&dataSymbols);

        ASTNodeCodeGen astCodeGen(builder, resolver, modelGenContext, modelData);

        const ListOfReactions *reactions = model->getListOfReactions();
        for (uint i = 0; i < reactions->size(); ++i) {
            const Reaction *reaction = reactions->get(i);

            const ListOfSpeciesReferences *products =
                    reaction->getListOfProducts();

            for (uint j = 0; j < products->size(); ++j) {
                const SpeciesReference *p = (const SpeciesReference *) products->get(j);

                if (p->isSetId() && p->getId().length() > 0 &&
                    !isConstantSpeciesReference(p)) {

                    if (dataSymbols.isBoundarySpecies(p->getSpecies())) {
                        // boundary species have no stoichiometry-matrix row,
                        // so there's no cell to resync here -- p's value is
                        // handled generically as a global parameter instead.
                        continue;
                    }

                    rrLog(Logger::LOG_INFORMATION) <<
                                                 "generating update code for non-constant species "
                                                 "reference product " << p->getId();

                    Value *value = 0;

                    if (dataSymbols.hasAssignmentRule(p->getId())
                        || dataSymbols.hasRateRule(p->getId())) {
                        value = resolver.loadSymbolValue(p->getId());
                    } else if (p->isSetStoichiometryMath()) {
                        const StoichiometryMath *sm = p->getStoichiometryMath();
                        value = astCodeGen.codeGenDouble(sm->getMath());
                    } else {
                        rrLog(Logger::LOG_WARNING) << "species reference "
                                                 << p->getId() << " has been determined to be "
                                                                  "non-constant, but it has no rules or MathML, so"
                                                                  " no update code will be generated";
                        continue;
                    }

                    assert(value && "value for species reference stoichiometry is 0");

                    const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                        dataSymbolsPtr->getNamedSpeciesReferenceInfo(p->getId());

                    if (info.type == LLVMModelDataSymbols::SpeciesReferenceType::MultiSpeciesReference) {
                        codeGenMultiSpeciesReferenceUpdate(mdbuilder, info, p->getId(), value, false);
                    } else {
                        mdbuilder.createStoichiometryStore(info.row, info.column,
                                                           value, p->getId());
                    }
                }
            }


            const ListOfSpeciesReferences *reactants =
                    reaction->getListOfReactants();

            for (uint j = 0; j < reactants->size(); ++j) {
                const SpeciesReference *r = (const SpeciesReference *) reactants->get(j);

                if (r->isSetId() && r->getId().length() > 0
                    && !isConstantSpeciesReference(r)) {

                    if (dataSymbols.isBoundarySpecies(r->getSpecies())) {
                        // see matching comment in the products loop above.
                        continue;
                    }

                    rrLog(Logger::LOG_INFORMATION) <<
                                                 "generating update code for non-constant species "
                                                 "reference reactant " << r->getId();

                    Value *value = 0;

                    if (dataSymbols.hasAssignmentRule(r->getId())
                        || dataSymbols.hasRateRule(r->getId())) {
                        value = resolver.loadSymbolValue(r->getId());
                    } else if (r->isSetStoichiometryMath()) {
                        const StoichiometryMath *sm = r->getStoichiometryMath();
                        value = astCodeGen.codeGenDouble(sm->getMath());
                    } else {
                        rrLog(Logger::LOG_WARNING) << "species reference "
                                                 << r->getId() << " has been determined to be "
                                                                  "non-constant, but it has no rules or MathML, so"
                                                                  " no update code will be generated";
                        continue;
                    }

                    assert(value && "value for species reference stoichiometry is 0");

                    const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                        dataSymbolsPtr->getNamedSpeciesReferenceInfo(r->getId());

                    if (info.type == LLVMModelDataSymbols::SpeciesReferenceType::MultiSpeciesReference) {
                        codeGenMultiSpeciesReferenceUpdate(mdbuilder, info, r->getId(), value, true);
                    } 
                    else {
                        // reactants are consumed, so they get a negative stoichiometry
                        Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
                        negOne->setName("neg_one");
                        value = builder.CreateFMul(negOne, value, "neg_" + r->getId());

                        mdbuilder.createStoichiometryStore(info.row, info.column, value,
                                                           r->getId());
                    }
                }
            }
        }

        builder.CreateRetVoid();

        return verifyFunction();
    }

    bool EvalVolatileStoichCodeGen::isConstantSpeciesReference(
            const libsbml::SimpleSpeciesReference *ref) const {
        const SpeciesReference *s = NULL;
        if (ref->getTypeCode() == SBML_SPECIES_REFERENCE) {
            s = static_cast<const SpeciesReference *>(ref);
        } else {
            return false;
        }

        if (ref->getLevel() >= 3 && ref->getVersion() >= 1) {
            return s->getConstant();
        } else if (dataSymbols.hasRateRule(s->getId())
                   || dataSymbols.hasAssignmentRule(s->getId())) {
            return false;
        } else if (s->isSetStoichiometryMath()) {
            return isConstantASTNode(s->getStoichiometryMath()->getMath());
        }
        return true;
    }

    template<typename type>
    static bool isSetConstant(const SBase *e, bool &isSetConst) {
        isSetConst = false;
        if (!e) {
            return false;
        }
        switch (e->getTypeCode()) {
            case SBML_SPECIES: {
                const Species *species = static_cast<const Species *>(e);
                isSetConst = species->getConstant();
                break;
            }
            case SBML_COMPARTMENT: {
                const Compartment *comp = static_cast<const Compartment *>(e);
                isSetConst = comp->getConstant();
                break;
            }
            case SBML_PARAMETER: {
                const Parameter *param = static_cast<const Parameter *>(e);
                isSetConst = param->getConstant();
                break;
            }
            case SBML_LOCAL_PARAMETER: {
                const LocalParameter *lp = static_cast<const LocalParameter *>(e);
                isSetConst = lp->getConstant();
                break;
            }
            case SBML_SPECIES_REFERENCE: {
                const SpeciesReference *sr = static_cast<const SpeciesReference *>(e);
                isSetConst = sr->getConstant();
                break;
            }
        }
        return true;
    }

    bool EvalVolatileStoichCodeGen::isConstantASTNode(const ASTNode *ast) const {
        const uint n = ast->getNumChildren();
        if (n > 0) {
            for (uint i = 0; i < n; ++i) {
                if (!isConstantASTNode(ast->getChild(i))) {
                    return false;
                }
            }
            return true;
        } else if (ast->isName()) {
            const Species* species = model->getSpecies(ast->getName());
            if (species) {
                return species->getConstant();
            }
            
            const Parameter* param = model->getParameter(ast->getName());
            if (param) {
                return param->getConstant();
            }

            const Compartment* comp = model->getCompartment(ast->getName());
            if (comp) {
                return comp->getConstant();
            }

            const ListOfReactions* lor = model->getListOfReactions();
            const SBase *element = const_cast<ListOfReactions *>(lor)->getElementBySId(
                    ast->getName());
            bool result;
            if (isSetConstant<SpeciesReference>(element, result)) {
                return result;
            } else {
                return false;
            }
        }
        return true;
    }

    void EvalVolatileStoichCodeGen::codeGenMultiSpeciesReferenceUpdate(
            ModelDataIRBuilder& mdbuilder,
            const LLVMModelDataSymbols::SpeciesReferenceInfo& info,
            const std::string& id, llvm::Value* rawValue, bool isReactant)
    {
        int slot = dataSymbols.getMultiSpeciesReferenceIndex(id);
        assert(slot >= 0 && "MultiSpeciesReference id missing from multiSpeciesReferenceMap");

        Value *stoichEP = mdbuilder.createGEP(Stoichiometry);
        Value *stoich = builder.CreateLoad(stoichEP->getType()->getPointerElementType(), stoichEP, "stoichiometry");
        Value *rowVal = ConstantInt::get(Type::getInt32Ty(builder.getContext()), info.row, true);
        Value *colVal = ConstantInt::get(Type::getInt32Ty(builder.getContext()), info.column, true);
        Value *oldCell = ModelDataIRBuilder::createCSRMatrixGetNZ(builder, stoich, rowVal, colVal);

        Value *aliasGEP = mdbuilder.createGEP(MultiSpeciesReferencesAlias, static_cast<unsigned>(slot), id);
        Value *oldRaw = builder.CreateLoad(aliasGEP->getType()->getPointerElementType(), aliasGEP, id + "_old");

        Value *delta = builder.CreateFSub(rawValue, oldRaw, "delta_" + id);
        if (isReactant) {
            Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
            delta = builder.CreateFMul(negOne, delta, "neg_delta_" + id);
        }
        Value *newCell = builder.CreateFAdd(oldCell, delta, "new_cell_" + id);

        ModelDataIRBuilder::createCSRMatrixSetNZ(builder, stoich, rowVal, colVal, newCell, id);

        builder.CreateStore(rawValue, aliasGEP);
    }


} /* namespace rrllvm */

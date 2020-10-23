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
    using namespace std;

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
                    Log(Logger::LOG_INFORMATION) <<
                                                 "generating update code for non-constant species "
                                                 "reference product " << p->getId();

                    Value *value = 0;

                    if (dataSymbols.hasAssignmentRule(p->getId())
                        || dataSymbols.hasRateRule(p->getId())) {
                        value = resolver.loadSymbolValue(p->getId());
                    } else if (p->isSetStoichiometryMath()) {
                        const StoichiometryMath *sm = p->getStoichiometryMath();
                        value = astCodeGen.codeGen(sm->getMath());
                    } else {
                        Log(Logger::LOG_WARNING) << "species reference "
                                                 << p->getId() << " has been determined to be "
                                                                  "non-constant, but it has no rules or MathML, so"
                                                                  " no update code will be generated";
                        continue;
                    }

                    assert(value && "value for species reference stoichiometry is 0");

                    const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                            dataSymbols.getNamedSpeciesReferenceInfo(p->getId());

                    mdbuilder.createStoichiometryStore(info.row, info.column,
                                                       value, p->getId());
                }
            }


            const ListOfSpeciesReferences *reactants =
                    reaction->getListOfReactants();

            for (uint j = 0; j < reactants->size(); ++j) {
                const SpeciesReference *r =
                        (const SpeciesReference *) reactants->get(j);

                if (r->isSetId() && r->getId().length() > 0
                    && !isConstantSpeciesReference(r)) {
                    Log(Logger::LOG_INFORMATION) <<
                                                 "generating update code for non-constant species "
                                                 "reference reactant " << r->getId();

                    const StoichiometryMath *sm = r->getStoichiometryMath();
                    assert(sm);

                    Value *value = astCodeGen.codeGen(sm->getMath());

                    // reactants are consumed, so they get a negative stoichiometry
                    Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
                    negOne->setName("neg_one");
                    value = builder.CreateFMul(negOne, value, "neg_" + r->getId());

                    const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                            dataSymbols.getNamedSpeciesReferenceInfo(r->getId());

                    mdbuilder.createStoichiometryStore(info.row, info.column, value,
                                                       r->getId());
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
            const SBase *element = const_cast<Model *>(model)->getElementBySId(
                    ast->getName());

            bool result;

            if (isSetConstant<Parameter>(element, result)) {
                return result;
            } else if (isSetConstant<Compartment>(element, result)) {
                return result;
            } else if (isSetConstant<Species>(element, result)) {
                return result;
            } else if (isSetConstant<SpeciesReference>(element, result)) {
                return result;
            } else {
                return false;
            }
        }
        return true;
    }


} /* namespace rrllvm */

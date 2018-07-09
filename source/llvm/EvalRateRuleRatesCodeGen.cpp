/*
 * rrLLVMEvalRateRuleRatesCodeGen.cpp
 *
 *  Created on: Aug 2, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EvalRateRuleRatesCodeGen.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "ASTNodeFactory.h"
#include "ModelDataSymbolResolver.h"
#include "rrLogger.h"
#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <Poco/Logger.h>


using namespace libsbml;
using namespace llvm;
using namespace std;


namespace rrllvm
{

const char* EvalRateRuleRatesCodeGen::FunctionName = "evalRateRuleRates";

EvalRateRuleRatesCodeGen::EvalRateRuleRatesCodeGen(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<EvalRateRuleRates_FunctionPtr>(mgc)
{
}

EvalRateRuleRatesCodeGen::~EvalRateRuleRatesCodeGen()
{
}

Value* EvalRateRuleRatesCodeGen::codeGen()
{
    Value *modelData = 0;

    codeGenVoidModelDataHeader(FunctionName, modelData);

    ModelDataLoadSymbolResolver resolver(modelData, modelGenContext);
    ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);
    ASTNodeCodeGen astCodeGen(builder, resolver);
    ASTNodeFactory nodes;

    // iterate through all of the reaction, and generate code based on thier
    // kinetic rules.

    model->getListOfRules();
    const ListOfRules *rules = model->getListOfRules();

    for (int i = 0; i < rules->size(); ++i)
    {
        const RateRule *rateRule = dynamic_cast<const RateRule*>(rules->get(i));
        const ASTNode *math = 0;

        if (rateRule)
        {
            // check if this rate rule applies to species, we only deal with
            // amounts and rates of change of amounts, so need to convert
            // accordignly
            const Species *species = dynamic_cast<const Species*>(
                    const_cast<Model*>(model)->getElementBySId(
                            rateRule->getVariable()));

            if (species)
            {
                if (!species->getHasOnlySubstanceUnits())
                {
                    // product rule, need to check if we have a rate rule for the
                    // species compartment.
                    const RateRule *compRateRule = dynamic_cast<const RateRule*>(
                            rules->get(species->getCompartment()));
                    if (compRateRule)
                    {
                        Log(Logger::LOG_DEBUG) << "species " << species->getId()
                                << " is a concentration with time dependent volume, "
                                "converting conc rate to amt rate using product rule";
                        ASTNode *dcdt = new ASTNode(*rateRule->getMath());
                        ASTNode *v = new ASTNode(AST_NAME);
                        v->setName(species->getCompartment().c_str());

                        ASTNode *dvdt = new ASTNode(*compRateRule->getMath());
                        ASTNode *c = new ASTNode(AST_NAME);
                        c->setName(species->getId().c_str());

                        ASTNode *l = new ASTNode(AST_TIMES);
                        l->addChild(dcdt);
                        l->addChild(c);

                        ASTNode *r = new ASTNode(AST_TIMES);
                        r->addChild(dvdt);
                        r->addChild(v);

                        ASTNode *plus = nodes.create(AST_PLUS);
                        plus->addChild(l);
                        plus->addChild(r);

                        math = plus;
                    }
                    else
                    {
                        Log(Logger::LOG_DEBUG) << "species " << species->getId()
                                << " is a concentration with constant volume, "
                                "converting conc rate to amt rate const vol mul";

                        ASTNode *dcdt = new ASTNode(*rateRule->getMath());
                        ASTNode *v = new ASTNode(AST_NAME);
                        v->setName(species->getCompartment().c_str());

                        ASTNode *times = nodes.create(AST_TIMES);
                        times->addChild(dcdt);
                        times->addChild(v);

                        math = times;
                    }
                }
                else
                {
                    Log(Logger::LOG_DEBUG) << "species " << species->getId() <<
                            " is an amount, creating straight rate rule";
                    math = rateRule->getMath();
                }
            }
            else
            {
                math = rateRule->getMath();
            }
            assert(math);
            Value *value = astCodeGen.codeGen(math);

            mdbuilder.createRateRuleRateStore(rateRule->getVariable(), value);
        }
    }

    builder.CreateRetVoid();

    return verifyFunction();
}

} /* namespace rr */

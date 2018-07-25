/*
 * rrLLVMModelDataTermSymbolResolver.cpp
 *
 *  Created on: Jul 25, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ModelDataSymbolResolver.h"
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "FunctionResolver.h"
#include "KineticLawParameterResolver.h"
#include "rrLogger.h"
#include <sbml/Model.h>

using namespace std;
using namespace libsbml;
using namespace llvm;

using rr::Logger;

namespace rrllvm
{

ModelDataLoadSymbolResolver::ModelDataLoadSymbolResolver(llvm::Value *modelData,
        const ModelGeneratorContext& ctx) :
            LoadSymbolResolverBase(ctx),
            modelData(modelData)
{
}

ModelDataStoreSymbolResolver::ModelDataStoreSymbolResolver(llvm::Value *modelData,
        const libsbml::Model *model,
        const LLVMModelSymbols &modelSymbols,
        const LLVMModelDataSymbols &modelDataSymbols,
        llvm::IRBuilder<> &builder,
        LoadSymbolResolver &resolver) :
            modelData(modelData),
            model(model),
            modelSymbols(modelSymbols),
            modelDataSymbols(modelDataSymbols),
            builder(builder),
            resolver(resolver)
{
}

//string ModelDataLoadSymbolResolver::decodeArraySymbol(const string& id) const
//{
//	size_t found = id.find_first_of("-");
//	string arrId = id.substr(0, found);
//	return arrId;
//}

llvm::Value* ModelDataLoadSymbolResolver::loadSymbolValue(
        const std::string& symbol,
        const llvm::ArrayRef<llvm::Value*>& args)
{
    {
        Value* cachedValue = cacheValue(symbol, args);
        if(cachedValue) return cachedValue;
    }

    ModelDataIRBuilder mdbuilder(modelData, modelDataSymbols,
            builder);

    /*************************************************************************/
    /* time */
    /*************************************************************************/
    if (symbol.compare(SBML_TIME_SYMBOL) == 0)
    {
        Value *timeEP = mdbuilder.createGEP(Time);
        Value *time = builder.CreateLoad(timeEP, SBML_TIME_SYMBOL);
        return cacheValue(symbol, args, time);
    }

    /*************************************************************************/
    /* Function */
    /*************************************************************************/
    {
        Value *funcVal =
            FunctionResolver(*this, modelData, modelGenContext).loadSymbolValue(symbol, args);
        if (funcVal)
        {
            return funcVal;
        }
    }

    /*************************************************************************/
    /* AssignmentRule */
    /*************************************************************************/
    {
		const string& arrayedSymbol = modelDataSymbols.decodeArrayId(symbol);
        SymbolForest::ConstIterator i = modelSymbols.getAssigmentRules().find(
                arrayedSymbol);
        if (i != modelSymbols.getAssigmentRules().end())
        {
            recursiveSymbolPush(symbol);
            Value* result = ASTNodeCodeGen(builder, *this).codeGen(i->second, symbol, args);
            recursiveSymbolPop();
            return cacheValue(symbol, args, result);
        }
    }

    /*************************************************************************/
    /* Species */
    /*************************************************************************/
    const Species *species = model->getSpecies(symbol);
    if (species)
    {
        Value *amt = 0;
        if (modelDataSymbols.isIndependentFloatingSpecies(symbol))
        {
            amt = mdbuilder.createFloatSpeciesAmtLoad(symbol, symbol + "_amt");
        }
        else if (modelDataSymbols.isIndependentBoundarySpecies(symbol))
        {
            amt = mdbuilder.createBoundSpeciesAmtLoad(symbol, symbol + "_amt");
        }
        else if(modelDataSymbols.hasRateRule(symbol))
        {
            amt = mdbuilder.createRateRuleValueLoad(symbol, symbol + "_amt");
        }
        else
        {
            string msg = string("the symbol ") + symbol + string(" appeared to "
                    "be a species, but it could not be found as an independent "
                    "species or rate rule");
            throw_llvm_exception(msg);
        }
        assert(amt);

        // now we have an amount, check to see if we need to convert to conc
        if (species->getHasOnlySubstanceUnits())
        {
            return cacheValue(symbol, args, amt);
        }
        else
        {
            // expect a concentration, need to convert amt to conc,
            // so we need to get the compartment its in, but these
            // can vary also...
            Value *comp = loadSymbolValue(species->getCompartment());
            return cacheValue(symbol, args, builder.CreateFDiv(amt, comp, symbol + "_conc"));
        }
    }

    if (modelDataSymbols.isIndependentCompartment(symbol))
    {
        return cacheValue(symbol, args, mdbuilder.createCompLoad(symbol));
    }

    if (modelDataSymbols.isIndependentGlobalParameter(symbol))
    {
        return cacheValue(symbol, args, mdbuilder.createGlobalParamLoad(symbol));
    }

    if (modelDataSymbols.hasRateRule(symbol))
    {
        // species conc / amt has already been taken care of at this point
        return cacheValue(symbol, args, mdbuilder.createRateRuleValueLoad(symbol));
    }

    if (modelDataSymbols.isNamedSpeciesReference(symbol))
    {
        const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                modelDataSymbols.getNamedSpeciesReferenceInfo(symbol);

        Value *value = mdbuilder.createStoichiometryLoad(info.row, info.column, symbol);

        if (info.type == LLVMModelDataSymbols::MultiReactantProduct)
        {
            string msg = "mutable stochiometry for species which appear "
                    "multiple times in a single reaction is not currently "
                    "supported, species reference id: ";
            msg += symbol;
            throw_llvm_exception(msg);
        }

        if (info.type == LLVMModelDataSymbols::Reactant)
        {
            // its consumed in the reaction, so has a negative in the stoich
            // matrix
            Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
            negOne->setName("neg_one");
            value = builder.CreateFMul(negOne, value, "neg_" + symbol);
        }

        return cacheValue(symbol, args, value);
    }

    /*************************************************************************/
    /* Reaction Rate */
    /*************************************************************************/
    const Reaction* reaction = model->getReaction(symbol);
    if (reaction)
    {
        return cacheValue(symbol, args, loadReactionRate(reaction));
    }


    string msg = "the symbol \'";
    msg += symbol;
    msg += "\' is not physically stored in the ModelData structure, "
            "it either does not exists or is defined by an assigment rule (hence it "
            "is not a terminal symbol)";

    throw_llvm_exception(msg);
    return 0;
}



llvm::Value* ModelDataStoreSymbolResolver::storeSymbolValue(
        const std::string& symbol, llvm::Value *value)
{
    assert(value);

    ModelDataIRBuilder mdbuilder(modelData, modelDataSymbols,
            builder);

    /*************************************************************************/
    /* AssignmentRule */
    /*************************************************************************/
    // can not store anything with an assigment rule, these are determined
    // by other independent elements.
    if (modelDataSymbols.hasAssignmentRule(symbol))
    {
        throw_llvm_exception("Attempt to store a value in symbol \""
                + symbol + "\" which is defined by an assignemnt rule");
    }


    /*************************************************************************/
    /* Species */
    /*************************************************************************/
    const Species *species = model->getSpecies(symbol);
    if (species)
    {
        Value *amt = 0;
        // only amounts are stored, convert to conc if required
        if (species->getHasOnlySubstanceUnits())
        {
            amt = value;
        }
        else
        {
            // have a conc, need to convert to amt
            Value *comp = resolver.loadSymbolValue(species->getCompartment());
            amt =  builder.CreateFMul(value, comp, symbol + "_amt");
        }

        assert(amt);

        // now look where we need to store it
        if (modelDataSymbols.isIndependentFloatingSpecies(symbol))
        {
            return mdbuilder.createFloatSpeciesAmtStore(symbol, amt);
        }
        else if (modelDataSymbols.isIndependentBoundarySpecies(symbol))
        {
            return mdbuilder.createBoundSpeciesAmtStore(symbol, amt);
        }
        else if(modelDataSymbols.hasRateRule(symbol))
        {
            return mdbuilder.createRateRuleValueStore(symbol, amt);
        }
        else
        {
            string msg = string("the symbol ") + symbol + string(" appeared to "
                    "be a species, but it could not be found as an independent "
                    "species or rate rule");
            throw_llvm_exception(msg);
            return 0;
        }
    }

    // at this point, we have already taken care of the species amount /
    // conc conversion, rest are just plain stores.

    if (modelDataSymbols.hasRateRule(symbol))
    {
        return mdbuilder.createRateRuleValueStore(symbol, value);
    }

    else if (modelDataSymbols.isIndependentCompartment(symbol))
    {
        return mdbuilder.createCompStore(symbol, value);
    }

    else if (modelDataSymbols.isIndependentGlobalParameter(symbol))
    {
        return mdbuilder.createGlobalParamStore(symbol, value);
    }

    else if (modelDataSymbols.isNamedSpeciesReference(symbol))
    {
        const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                modelDataSymbols.getNamedSpeciesReferenceInfo(symbol);

        if (info.type == LLVMModelDataSymbols::MultiReactantProduct)
        {
            string msg = "mutable stochiometry for species which appear "
                    "multiple times in a single reaction is not currently "
                    "supported, species reference id: ";
            msg += symbol;
            throw_llvm_exception(msg);
        }

        if (info.type == LLVMModelDataSymbols::Reactant)
        {
            // its consumed in the reaction, so has a negative in the stoich
            // matrix
            Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
            negOne->setName("neg_one");
            value = builder.CreateFMul(negOne, value, "neg_" + symbol);
        }

        return mdbuilder.createStoichiometryStore(info.row, info.column, value);
    }

    string msg = "The symbol \'";
    msg += symbol;
    msg += "\' is not physically stored in the ModelData structure, "
            "it either does not exists or is defined by an assigment rule (hence it "
            "is not a terminal symbol)";

    throw_llvm_exception(msg);
    return 0;
}

} /* namespace rr */



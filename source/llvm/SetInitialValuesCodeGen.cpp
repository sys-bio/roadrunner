/*
 * GetValuesCodeGen.cpp
 *
 *  Created on: Sept 16, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "SetInitialValuesCodeGen.h"
#include "LLVMException.h"
#include "ModelDataSymbolResolver.h"
#include "rrLogger.h"

#include <Poco/Logger.h>
#include <vector>
#include <algorithm>

using namespace llvm;
using namespace std;



namespace rrllvm
{


const char* SetFloatingSpeciesInitConcentrationCodeGen::FunctionName = "setFloatingSpeciesInitConcentrations";
const char* SetFloatingSpeciesInitConcentrationCodeGen::IndexArgName = "floatingSpeciesIndex";

SetFloatingSpeciesInitConcentrationCodeGen::SetFloatingSpeciesInitConcentrationCodeGen(
        const ModelGeneratorContext &mgc) :
        SetInitialValueCodeGenBase<SetFloatingSpeciesInitConcentrationCodeGen, false>(mgc)
{
}

StringIntVector SetFloatingSpeciesInitConcentrationCodeGen::getIds()
{
    std::vector<string> ids = dataSymbols.getFloatingSpeciesIds();
    StringIntVector result;

    for(StringVector::iterator i = ids.begin(); i != ids.end(); ++i)
    {
        if (dataSymbols.isIndependentInitFloatingSpecies(*i))
        {
            result.push_back(make_pair(*i, distance(ids.begin(), i)));
        }
    }
    return result;
}


const char* SetFloatingSpeciesInitAmountCodeGen::FunctionName = "setFloatingSpeciesInitAmounts";
const char* SetFloatingSpeciesInitAmountCodeGen::IndexArgName = "floatingSpeciesIndex";

SetFloatingSpeciesInitAmountCodeGen::SetFloatingSpeciesInitAmountCodeGen(
        const ModelGeneratorContext &mgc) :
        SetInitialValueCodeGenBase<SetFloatingSpeciesInitAmountCodeGen, true>(mgc)
{
}

StringIntVector SetFloatingSpeciesInitAmountCodeGen::getIds()
{
    std::vector<string> ids = dataSymbols.getFloatingSpeciesIds();
    StringIntVector result;

    for(StringVector::iterator i = ids.begin(); i != ids.end(); ++i)
    {
        if (dataSymbols.isIndependentInitFloatingSpecies(*i))
        {
            result.push_back(make_pair(*i, distance(ids.begin(), i)));
        }
    }
    return result;
}

const char* SetCompartmentInitVolumeCodeGen::FunctionName = "setCompartmentInitVolumes";
const char* SetCompartmentInitVolumeCodeGen::IndexArgName = "compartmentIndex";

SetCompartmentInitVolumeCodeGen::SetCompartmentInitVolumeCodeGen(
        const ModelGeneratorContext &mgc) :
        SetInitialValueCodeGenBase<SetCompartmentInitVolumeCodeGen, false>(mgc)
{
}

StringIntVector SetCompartmentInitVolumeCodeGen::getIds()
{
    std::vector<string> ids = dataSymbols.getCompartmentIds();
    StringIntVector result;

    for(StringVector::iterator i = ids.begin(); i != ids.end(); ++i)
    {
        if (dataSymbols.isIndependentInitCompartment(*i))
        {
            result.push_back(make_pair(*i, distance(ids.begin(), i)));
        }
    }
    return result;
}


const char* SetGlobalParameterInitValueCodeGen::FunctionName = "setGlobalParameterInitValue";
const char* SetGlobalParameterInitValueCodeGen::IndexArgName = "globalParameterIndex";

SetGlobalParameterInitValueCodeGen::SetGlobalParameterInitValueCodeGen(
        const ModelGeneratorContext &mgc) :
        SetInitialValueCodeGenBase<SetGlobalParameterInitValueCodeGen, false>(mgc)
{
}

StringIntVector SetGlobalParameterInitValueCodeGen::getIds()
{
    std::vector<string> ids = dataSymbols.getGlobalParameterIds();
    StringIntVector result;

    for(StringVector::iterator i = ids.begin(); i != ids.end(); ++i)
    {
        if (dataSymbols.isIndependentInitGlobalParameter(*i))
        {
            result.push_back(make_pair(*i, distance(ids.begin(), i)));
        }
    }
    return result;
}


} /* namespace rr */


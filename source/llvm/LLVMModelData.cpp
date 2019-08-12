/*
 * rrLLVMModelData.cpp
 *
 *  Created on: Aug 8, 2013
 *      Author: andy
 */

#pragma hdrstop
#include "LLVMModelData.h"

#include <stdlib.h>
#include <string.h>
#include "rrExecutableModel.h"
#include "rrSparse.h"
#include "Random.h"
#include <iomanip>
#include <iostream>
#include "rrStringUtils.h"

using namespace std;

static void dump_array(std::ostream &os, int n, const double *p)
{
    if (p)
    {
        os << setiosflags(ios::floatfield) << setprecision(8);
        os << '[';
        for (int i = 0; i < n; ++i)
        {
            os << p[i];
            if (i < n - 1)
            {
                os << ", ";
            }
        }
        os << ']' << endl;
    }
    else
    {
        os << "NULL" << endl;
    }
}


namespace rrllvm
{


std::ostream& operator <<(std::ostream& os, const LLVMModelData& data)
{
    os << "LLVMModelData:"                  << endl;
    os << "size: "                          << data.size << endl;
    os << "flags: "                         << data.flags << endl;
    os << "time: "                          << data.time << endl;
    os << "numIndFloatingSpecies: "         << data.numIndFloatingSpecies << endl;

    os << "numIndGlobalParameters: "        << data.numIndGlobalParameters << endl;
    os << "globalParameters: "              << endl;
    dump_array(os, data.numIndGlobalParameters, data.globalParametersAlias);

    os << "numReactions: "                  << data.numReactions << endl;
    os << "reactionRates: "                 << endl;
    dump_array(os, data.numReactions, data.reactionRatesAlias);

    os << "numRateRules: "                  << data.numRateRules << endl;
    os << "rateRuleValues: "                << endl;
    dump_array(os, data.numRateRules, data.rateRuleValuesAlias);

    os << "floatingSpeciesAmounts: "        << endl;
    dump_array(os, data.numIndFloatingSpecies, data.floatingSpeciesAmountsAlias);

    os << "numIndBoundarySpecies: "         << data.numIndBoundarySpecies << endl;

    os << "boundarySpeciesAmounts:"         << endl;
    dump_array(os, data.numIndBoundarySpecies, data.boundarySpeciesAmountsAlias);

    os << "numIndCompartments: "            << data.numIndCompartments << endl;
    os << "compartmentVolumes:"             << endl;
    dump_array(os, data.numIndCompartments, data.compartmentVolumesAlias);

    os << "stoichiometry:"                  << endl;
    os << data.stoichiometry;


    os << "numInitFloatingSpecies: "        << data.numInitFloatingSpecies << endl;
    os << "initFloatingSpeciesAmounts: "    << endl;
    dump_array(os, data.numInitFloatingSpecies, data.initFloatingSpeciesAmountsAlias);

    os << "numInitCompartments: "           << data.numInitCompartments << endl;
    os << "initCompartmentVolumes:"         << endl;
    dump_array(os, data.numInitCompartments, data.initCompartmentVolumesAlias);

    os << "numInitGlobalParameters: "       << data.numInitGlobalParameters << endl;
    os << "initGlobalParameters: "          << endl;
    dump_array(os, data.numInitGlobalParameters, data.initGlobalParametersAlias);


    return os;
}

void LLVMModelData_save(LLVMModelData *data, std::ostream& out) 
{
	//Counts and size variables
	rr::saveBinary(out, data->size);
	rr::saveBinary(out, data->flags);
	rr::saveBinary(out, data->time);
	rr::saveBinary(out, data->numIndCompartments);
	rr::saveBinary(out, data->numIndFloatingSpecies);
	rr::saveBinary(out, data->numIndBoundarySpecies);
	rr::saveBinary(out, data->numIndGlobalParameters);
	rr::saveBinary(out, data->numRateRules);
	rr::saveBinary(out, data->numReactions);
	rr::saveBinary(out, data->numInitCompartments);
	rr::saveBinary(out, data->numInitFloatingSpecies);
	rr::saveBinary(out, data->numInitBoundarySpecies);
	rr::saveBinary(out, data->numInitGlobalParameters);
    
	rr::saveBinary(out, data->numEvents);
	rr::saveBinary(out, data->stateVectorSize);
	//Save the stoichiometry matrix
	rr::csr_matrix_dump_binary(data->stoichiometry, out);

	//We do not need to save random because LLVMExecutableModel will make a new one if it is null
    
	//We do not need to save stateVector as it is allocated by the integrator

	//rateRuleRates is only valid during an evalModel call, we don't need to save it

	//Alias pointer offsets
	unsigned compartmentVolumesAliasOffset = data->compartmentVolumesAlias - data->data;
	rr::saveBinary(out, compartmentVolumesAliasOffset);
    
	unsigned initCompartmentVolumesAliasOffset = data->initCompartmentVolumesAlias - data->data;
	rr::saveBinary(out, initCompartmentVolumesAliasOffset);

	unsigned initFloatingSpeciesAmountAliasOffset = data->initFloatingSpeciesAmountsAlias - data->data;
	rr::saveBinary(out, initFloatingSpeciesAmountAliasOffset);

	unsigned boundarySpeciesAmountAliasOffset = data->boundarySpeciesAmountsAlias - data->data;
	rr::saveBinary(out, boundarySpeciesAmountAliasOffset); 

	unsigned initBoundarySpeciesAmountsAliasOffset = data->initBoundarySpeciesAmountsAlias - data->data;
	rr::saveBinary(out, initBoundarySpeciesAmountsAliasOffset); 

	unsigned globalParametersAliasOffset = data->globalParametersAlias - data->data;
	rr::saveBinary(out, globalParametersAliasOffset); 

	unsigned initGlobalParametersAliasOffset = data->initGlobalParametersAlias - data->data;
	rr::saveBinary(out, initGlobalParametersAliasOffset); 

	unsigned reactionRatesAliasOffset = data->reactionRatesAlias - data->data;
	rr::saveBinary(out, reactionRatesAliasOffset); 

	unsigned rateRuleValuesAliasOffset = data->rateRuleValuesAlias - data->data;
	rr::saveBinary(out, rateRuleValuesAliasOffset); 

	unsigned floatingSpeciesAmountsAliasOffset = data->floatingSpeciesAmountsAlias - data->data;
	rr::saveBinary(out, floatingSpeciesAmountsAliasOffset); 

	//save the data itself
	//the size is the sum of the 10 unsigned integers at the top of LLVMModelData
	unsigned dataSize = data->numIndCompartments + data->numIndFloatingSpecies + data->numIndBoundarySpecies + 
		                data->numIndGlobalParameters + data->numRateRules + data->numReactions + data->numInitCompartments + data->numInitFloatingSpecies + 
		                data->numInitBoundarySpecies + data->numInitGlobalParameters;

	out.write((char*)(data->data), dataSize*sizeof(double));
}

/*
* Allocates and returns a pointer to a new LLVMModelData object
* based on the save data fed by in
*/
LLVMModelData* LLVMModelData_from_save(std::istream& in)
{
	//Counts and size variables
	unsigned size;
	rr::loadBinary(in, size);

	LLVMModelData *data = (LLVMModelData*)calloc(size, sizeof(unsigned char));

	data->size = size;
	rr::loadBinary(in, data->flags);
	rr::loadBinary(in, data->time);
	rr::loadBinary(in, data->numIndCompartments);
	rr::loadBinary(in, data->numIndFloatingSpecies);
	rr::loadBinary(in, data->numIndBoundarySpecies);
	rr::loadBinary(in, data->numIndGlobalParameters);
	rr::loadBinary(in, data->numRateRules);
	rr::loadBinary(in, data->numReactions);
	rr::loadBinary(in, data->numInitCompartments);
	rr::loadBinary(in, data->numInitFloatingSpecies);
	rr::loadBinary(in, data->numInitBoundarySpecies);
	rr::loadBinary(in, data->numInitGlobalParameters);

	rr::loadBinary(in, data->numEvents);
	rr::loadBinary(in, data->stateVectorSize);
    
	//Load the stoichiometry matrix
	data->stoichiometry = rr::csr_matrix_new_from_binary(in);

	//Alias pointer offsets

	unsigned compartmentVolumesAliasOffset;
	rr::loadBinary(in, compartmentVolumesAliasOffset);
	data->compartmentVolumesAlias = data->data + compartmentVolumesAliasOffset;

	unsigned initCompartmentVolumesAliasOffset;
	rr::loadBinary(in, initCompartmentVolumesAliasOffset);
	data->initCompartmentVolumesAlias = data->data + initCompartmentVolumesAliasOffset;

	unsigned initFloatingSpeciesAmountsAliasOffset;
	rr::loadBinary(in, initFloatingSpeciesAmountsAliasOffset);
	data->initFloatingSpeciesAmountsAlias = data->data + initFloatingSpeciesAmountsAliasOffset;

	unsigned boundarySpeciesAmountsAliasOffset;
	rr::loadBinary(in, boundarySpeciesAmountsAliasOffset);
	data->boundarySpeciesAmountsAlias = data->data + boundarySpeciesAmountsAliasOffset;

	unsigned initBoundarySpeciesAmountsAliasOffset;
	rr::loadBinary(in, initBoundarySpeciesAmountsAliasOffset);
	data->initBoundarySpeciesAmountsAlias = data->data + initBoundarySpeciesAmountsAliasOffset;

	unsigned globalParametersAliasOffset;
	rr::loadBinary(in, globalParametersAliasOffset);
	data->globalParametersAlias = data->data + globalParametersAliasOffset;

	unsigned initGlobalParametersAliasOffset;
	rr::loadBinary(in, initGlobalParametersAliasOffset);
	data->initGlobalParametersAlias = data->data + initGlobalParametersAliasOffset;

	unsigned reactionRatesAliasOffset;
	rr::loadBinary(in, reactionRatesAliasOffset);
	data->reactionRatesAlias = data->data + reactionRatesAliasOffset;

	unsigned rateRuleValuesAliasOffset;
	rr::loadBinary(in, rateRuleValuesAliasOffset);
	data->rateRuleValuesAlias = data->data + rateRuleValuesAliasOffset;

	unsigned floatingSpeciesAmountsAliasOffset;
	rr::loadBinary(in, floatingSpeciesAmountsAliasOffset);
	data->floatingSpeciesAmountsAlias = data->data + floatingSpeciesAmountsAliasOffset;

	//save the data itself
	//the size is the sum of the 10 unsigned integers at the top of LLVMModelData
	unsigned dataSize = data->numIndCompartments + data->numIndFloatingSpecies + data->numIndBoundarySpecies + 
		                data->numIndGlobalParameters + data->numRateRules + data->numReactions + data->numInitCompartments + data->numInitFloatingSpecies + 
		                data->numInitBoundarySpecies + data->numInitGlobalParameters;
	if (dataSize*sizeof(double) + sizeof(LLVMModelData) != size) {
		size = dataSize + sizeof(LLVMModelData);
	}
	in.read((char*)(data->data), dataSize*sizeof(double));
	return data;
}

void  LLVMModelData_free(LLVMModelData *data)
{
    if (data)
    {
        csr_matrix_delete(data->stoichiometry);
        delete data->random;
        ::free(data);
    }
}

} // namespace rr






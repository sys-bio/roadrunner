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
	out.write((char*)&(data->size), sizeof data->size);
	out.write((char*)&(data->flags), sizeof data->flags);
	out.write((char*)&(data->time), sizeof data->time);
	out.write((char*)&(data->numIndCompartments), sizeof data->numIndCompartments);
	out.write((char*)&(data->numIndFloatingSpecies), sizeof data->numIndFloatingSpecies);
	out.write((char*)&(data->numIndBoundarySpecies), sizeof data->numIndBoundarySpecies);
	out.write((char*)&(data->numIndGlobalParameters), sizeof data->numIndGlobalParameters);
	out.write((char*)&(data->numRateRules), sizeof data->numRateRules);
	out.write((char*)&(data->numReactions), sizeof data->numReactions);
	out.write((char*)&(data->numInitCompartments), sizeof data->numInitCompartments);
	out.write((char*)&(data->numInitFloatingSpecies), sizeof data->numInitFloatingSpecies);
	out.write((char*)&(data->numInitBoundarySpecies), sizeof data->numInitBoundarySpecies);
	out.write((char*)&(data->numInitGlobalParameters), sizeof data->numInitGlobalParameters);

	out.write((char*)&(data->numEvents), sizeof data->numEvents);
	out.write((char*)&(data->stateVectorSize), sizeof data->stateVectorSize);
    
	//Save the stoichiometry matrix
	rr::csr_matrix_dump_binary(data->stoichiometry, out);

    //We do not need to save random because LLVMExecutableModel will make a new one if it is null
    
	//It appears that stateVector and stateVectorRate are entirely unused

	//rateRuleRates is only valid during an evalModel call, we don't need to save it

	//Alias pointer offsets
	unsigned compartmentVolumesAliasOffset = data->compartmentVolumesAlias - data->data;
	out.write((char*)&(compartmentVolumesAliasOffset), sizeof(unsigned));

	unsigned initCompartmentVolumesAliasOffset = data->initCompartmentVolumesAlias - data->data;
	out.write((char*)&(initCompartmentVolumesAliasOffset), sizeof(unsigned));

	unsigned initFloatingSpeciesAmountAliasOffset = data->initFloatingSpeciesAmountsAlias - data->data;
	out.write((char*)&(initFloatingSpeciesAmountAliasOffset), sizeof(unsigned));

	unsigned boundarySpeciesAmountAliasOffset = data->boundarySpeciesAmountsAlias - data->data;
	out.write((char*)&(boundarySpeciesAmountAliasOffset), sizeof(unsigned));

	unsigned initBoundarySpeciesAmountsAliasOffset = data->initBoundarySpeciesAmountsAlias - data->data;
	out.write((char*)&(initBoundarySpeciesAmountsAliasOffset), sizeof(unsigned));

	unsigned globalParametersAliasOffset = data->globalParametersAlias - data->data;
	out.write((char*)&(globalParametersAliasOffset), sizeof(unsigned));

	unsigned initGlobalParametersAliasOffset = data->initGlobalParametersAlias - data->data;
	out.write((char*)&(initGlobalParametersAliasOffset), sizeof(unsigned));

	unsigned reactionRatesAliasOffset = data->reactionRatesAlias - data->data;
	out.write((char*)&(reactionRatesAliasOffset), sizeof(unsigned));

	unsigned rateRuleValuesAliasOffset = data->rateRuleValuesAlias - data->data;
	out.write((char*)&(rateRuleValuesAliasOffset), sizeof(unsigned));

	unsigned floatingSpeciesAmountsAliasOffset = data->floatingSpeciesAmountsAlias - data->data;
	out.write((char*)&(floatingSpeciesAmountsAliasOffset), sizeof(unsigned));

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
	in.read((char*)&(size), sizeof(unsigned));

	LLVMModelData *data = (LLVMModelData*)calloc(size, sizeof(unsigned char));

	data->size = size;
	in.read((char*)&(data->flags), sizeof data->flags);
	in.read((char*)&(data->time), sizeof data->time);
	in.read((char*)&(data->numIndCompartments), sizeof data->numIndCompartments);
	in.read((char*)&(data->numIndFloatingSpecies), sizeof data->numIndFloatingSpecies);
	in.read((char*)&(data->numIndBoundarySpecies), sizeof data->numIndBoundarySpecies);
	in.read((char*)&(data->numIndGlobalParameters), sizeof data->numIndGlobalParameters);
	in.read((char*)&(data->numRateRules), sizeof data->numRateRules);
	in.read((char*)&(data->numReactions), sizeof data->numReactions);
	in.read((char*)&(data->numInitCompartments), sizeof data->numInitCompartments);
	in.read((char*)&(data->numInitFloatingSpecies), sizeof data->numInitFloatingSpecies);
	in.read((char*)&(data->numInitBoundarySpecies), sizeof data->numInitBoundarySpecies);
	in.read((char*)&(data->numInitGlobalParameters), sizeof data->numInitGlobalParameters);

	in.read((char*)&(data->numEvents), sizeof data->numEvents);
	in.read((char*)&(data->stateVectorSize), sizeof data->stateVectorSize);
    
	//Load the stoichiometry matrix
	data->stoichiometry = rr::csr_matrix_new_from_binary(in);

	//Alias pointer offsets

	unsigned compartmentVolumesAliasOffset;
	in.read((char*)&(compartmentVolumesAliasOffset), sizeof(unsigned));
	data->compartmentVolumesAlias = data->data + compartmentVolumesAliasOffset;

	unsigned initCompartmentVolumesAliasOffset;
	in.read((char*)&(initCompartmentVolumesAliasOffset), sizeof(unsigned));
	data->initCompartmentVolumesAlias = data->data + initCompartmentVolumesAliasOffset;

	unsigned initFloatingSpeciesAmountsAliasOffset;
	in.read((char*)&(initFloatingSpeciesAmountsAliasOffset), sizeof(unsigned));
	data->initFloatingSpeciesAmountsAlias = data->data + initFloatingSpeciesAmountsAliasOffset;

	unsigned boundarySpeciesAmountsAliasOffset;
	in.read((char*)&(boundarySpeciesAmountsAliasOffset), sizeof(unsigned));
	data->boundarySpeciesAmountsAlias = data->data + boundarySpeciesAmountsAliasOffset;

	unsigned initBoundarySpeciesAmountsAliasOffset;
	in.read((char*)&(initBoundarySpeciesAmountsAliasOffset), sizeof(unsigned));
	data->initBoundarySpeciesAmountsAlias = data->data + initBoundarySpeciesAmountsAliasOffset;

	unsigned globalParametersAliasOffset;
	in.read((char*)&(globalParametersAliasOffset), sizeof(unsigned));
	data->globalParametersAlias = data->data + globalParametersAliasOffset;

	unsigned initGlobalParametersAliasOffset;
	in.read((char*)&(initGlobalParametersAliasOffset), sizeof(unsigned));
	data->initGlobalParametersAlias = data->data + initGlobalParametersAliasOffset;

	unsigned reactionRatesAliasOffset;
	in.read((char*)&(reactionRatesAliasOffset), sizeof(unsigned));
	data->reactionRatesAlias = data->data + reactionRatesAliasOffset;

	unsigned rateRuleValuesAliasOffset;
	in.read((char*)&(rateRuleValuesAliasOffset), sizeof(unsigned));
	data->rateRuleValuesAlias = data->data + rateRuleValuesAliasOffset;

	unsigned floatingSpeciesAmountsAliasOffset;
	in.read((char*)&(floatingSpeciesAmountsAliasOffset), sizeof(unsigned));
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






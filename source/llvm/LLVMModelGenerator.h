/*
 * LLVMModelGenerator.h
 *
 * Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#ifndef rrLLVMModelGeneratorH
#define rrLLVMModelGeneratorH

#include "rrRoadRunnerOptions.h"
#include "LLVMCompiler.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"

#include "LLVMExecutableModel.h"
#include "ModelGeneratorContext.h"

namespace rrllvm
{

/**
 * General concepts:
 *
 * SBML defines chemical species.
 *
 * Floating Species: species whose value varies over time, i.e. they can 'float'
 * Boundary Species: boundary condition species, values are fixed to initial conditions.
 */
class RR_DECLSPEC LLVMModelGenerator
{
public:

	/**
	 * Create an executable model from an sbml std::string
	 */
    static rr::ExecutableModel *createModel(const std::string& sbml, std::uint32_t options);

	/**
	 * Regenerate an executable model from an existing sbml document i.e. generate the model normally from the SBML document,
	 * but copy the values of the floating species (initial and current), boundary species, compartments (initial and current),
	 * global parameters (initial and current) that are in both oldModel and doc, as well as time 
	 * from the old model into the one returned
	 */
	static rr::ExecutableModel *regenerateModel(rr::ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options);

};

} /* namespace rr */
#endif /* rrLLVMModelGeneratorH */

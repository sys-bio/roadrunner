/*
 * ExecutableModelFactory.h
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#ifndef EXECUTABLEMODELFACTORY_H_
#define EXECUTABLEMODELFACTORY_H_

#include "rrExecutableModel.h"
#include "Dictionary.h"
#include <string>
#include <iostream>


namespace rr
{

/**
 * A factory class to create ExecutableModel objects.
 */
class RR_DECLSPEC ExecutableModelFactory
{
public:
    /**
     * creates a NEW object which must be deleted by the caller.
     *
     * @param sbml: an sbml string
     * @param dict: a dictionary of options, this is typcally a LoadSBMLOptions object,
     * but it may be any dictionary.
     */
    static ExecutableModel *createModel(const std::string& sbml, const Dictionary* dict = 0);
	/*
	* Creates an executable model from a saved binary representation
	*
	* @param in: input stream with the representation of the model at the front
	* @param modelGeneratorOpt: which functions should be generated, see LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS
	* and LoadSBMLOptions::READ_ONLY
	*
	*/
	static ExecutableModel *createModel(std::istream& in, uint modelGeneratorOpt);

	/*
	* Regenerate an executable model from a saved sbml document
	*
	* @param doc: sbml document that generate the new executable model from
	* @param LoadSBMLOptions: load options
	*
	*/
	static ExecutableModel *regenerateModel(ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options);
};

} /* namespace rr */

#endif /* EXECUTABLEMODELFACTORY_H_ */

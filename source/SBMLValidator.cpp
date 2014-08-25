/*
 * SBMLValidator.cpp
 *
 *  Created on: Aug 24, 2014
 *      Author: andy
 */

#include "SBMLValidator.h"
#include "rrSBMLReader.h"

#include <sbml/SBMLDocument.h>
#include <sbml/SBMLReader.h>

#include <sbml/validator/ConsistencyValidator.h>
#include <sbml/validator/UnitConsistencyValidator.h>
#include <sbml/validator/IdentifierConsistencyValidator.h>
#include <sbml/validator/MathMLConsistencyValidator.h>
#include <sbml/validator/OverdeterminedValidator.h>
#include <sbml/validator/ModelingPracticeValidator.h>

#include "rrLogger.h"

#include <sstream>

using libsbml::SBMLDocument;
using libsbml::readSBMLFromString;
using std::string;

using namespace libsbml;


typedef std::list<SBMLError> SBMLErrorList;



namespace rr
{

struct RRValidator
{
    std::stringstream errors;
    SBMLDocument *doc;

    RRValidator() : doc(NULL) {}

    void load(const std::string& src) {
        string sbml = SBMLReader::read(src);
        doc =  readSBMLFromString (sbml.c_str());
    }

    ~RRValidator() {
        delete doc;
    }

    void validate(libsbml::Validator& val, const std::string& name) {
        val.init();

        int err = val.validate(*doc);

        if (err > 0) {

            errors << name << ":" << std::endl;

            for (SBMLErrorList::const_iterator i = val.getFailures().begin();
                    i != val.getFailures().end(); ++i) {
                i->print(errors);
            }
        }
    }
};


std::string validateSBML(const std::string src, unsigned opt)
{
    RRValidator v;

    v.load(src);

    if (opt & VALIDATE_GENERAL)
    {
        ConsistencyValidator val;
        v.validate(val, "VALIDATE_GENERAL");
    }

    if (opt & VALIDATE_UNITS)
    {
        UnitConsistencyValidator val;
        v.validate(val, "VALIDATE_UNITS");
    }

    if (opt & VALIDATE_IDENTIFIER)
    {
        IdentifierConsistencyValidator val;
        v.validate(val, "VALIDATE_IDENTIFIER");
    }

    if (opt & VALIDATE_MATHML)
    {
        MathMLConsistencyValidator val;
        v.validate(val, "VALIDATE_MATHML");
    }

    if (opt & VALIDATE_OVERDETERMINED)
    {
        OverdeterminedValidator val;
        v.validate(val, "VALIDATE_OVERDETERMINED");
    }

    if (opt & VALIDATE_MODELING_PRACTICE)
    {
        ModelingPracticeValidator val;
        v.validate(val, "VALIDATE_MODELING_PRACTICE");
    }

    return v.errors.str();
}

} /* namespace rr */

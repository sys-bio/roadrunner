// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * GPUSimModel.h
 *
 *  Created on: Aug 14, 2014
 *
 * Author: JKM
 */

#ifndef GPUSimModelH
#define GPUSimModelH

// == INCLUDES ================================================

#include "rrSelectionRecord.h"
#include "conservation/ConservedMoietyConverter.h"
#include "conservation/ConservationExtension.h"

#include <memory>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

/**
 * @author JKM
 * @brief Class that holds model data, symbols, and contextual information
 */
class RR_DECLSPEC GPUSimModel
{
public:

    GPUSimModel(std::string const &sbml, unsigned loadSBMLOptions);

protected:
    /// Returns the document access pointer
    libsbml::SBMLDocument* getDocument();

    /// Returns the document access pointer
    const libsbml::SBMLDocument* getDocument() const;

    typedef std::unique_ptr<conservation::ConservedMoietyConverter> ConservedMoietyConverterPtr;
    /**
     * the moiety converter, for the time being owns the
     * converted document.
     */
    ConservedMoietyConverterPtr moietyConverter;

    typedef std::unique_ptr<libsbml::SBMLDocument> SBMLDocumentPtr;
    SBMLDocumentPtr ownedDoc;
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */

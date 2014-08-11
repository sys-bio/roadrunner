/**
 * @file    FbcSBMLErrorTable.h
 * @brief   Represents SBML errors and other diagnostics
 * @author  Sarah Keating
 * 
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 * 
 * Copyright (C) 2013-2014 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *     3. University of Heidelberg, Heidelberg, Germany
 * 
 * Copyright (C) 2009-2013 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->
 */
 
#include <string>
#include <iostream>
#include <sstream>

#include <sbml/packages/fbc/validator/FbcSBMLError.h>


LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygenLibsbmlInternal */



static const packageErrorTableEntry fbcErrorTable[] =
{
  // 2010100
  { FbcUnknown, 
    "Unknown error from fbc",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "Unknown error from fbc",
    { ""
    }
  },

  // 2010101
  { FbcNSUndeclared, 
    "The fbc ns is not correctly declared",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "To conform to Version 1 of the Flux Balance Constraints "
    "package specification for SBML Level 3, an "
    "SBML document must declare the use of the following XML Namespace: "
    "'http://www.sbml.org/sbml/level3/version1/fbc/version1'",
    { "L3V1 Fbc V1 Section 3.1"
    }
  },

  // 2010102
  { FbcElementNotInNs, 
    "Element not in fbc namespace",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "Wherever they appear in an SBML document, "
    "elements and attributes from the Flux Balance Constraints "
    "package must be declared either implicitly or explicitly to be in the "
    "XML namespace 'http://www.sbml.org/sbml/level3/version1/fbc/version1'",
    { "L3V1 Fbc V1 Section 3.1"
    }
  },

    // 2010301
  { FbcDuplicateComponentId, 
  "Duplicate 'id' attribute value",
    LIBSBML_CAT_IDENTIFIER_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "(Extends validation rule #10301 in the SBML Level 3 Version 1 Core "
    "specification.) Within a <model> object"
    "the values of the attributes id and fbc:id on every instance of the "
    "following classes of objects must be unique across the set of all id "
    "and fbc:id attribute values of all such objects in a model: the "
    "model itself, plus all contained <functionDefinition>, <compartment>,"
    "<species>, <reaction>, <speciesReference>, <modifierSpeciesReference>, "
    "<event>, and <parameter> objects, plus the <fluxBound>, <objective> "
    "and <fluxObjective> "
    "objects defined by the Flux Balance Constraints package.", 
    { "L3V1 Fbc V1 Section 3.2"
    }
  },

    // 2010302
  { FbcSBMLSIdSyntax, 
    "Invalid 'id' attribute",
    LIBSBML_CAT_IDENTIFIER_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of a fbc:id attribute must always conform to the syntax "
    "of the SBML data type SId.", 
    { "L3V1 Fbc V1 Section 3.2"
    }
  },

  // 2020101
  { FbcAttributeRequiredMissing, 
    "Required fbc:required attribute on <sbml>",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "In all SBML documents using the Flux Balance Constraints "
    "package, the SBML object must include a value for the attribute "
    "'fbc:required.",
    { "L3V1 Core Section 4.1.2"
    }
  },

  // 2020102
  { FbcAttributeRequiredMustBeBoolean, 
    "The fbc:required attribute must be Boolean",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:required' on the SBML object must "
    "be of the data type Boolean.",
    { "L3V1 Core Section 4.1.2"
    }
  },

  // 2020103
  { FbcRequiredFalse, 
    "The fbc:required attribute must be 'false'",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:required' on the SBML object must "
    "be set to 'false'.",
    { "L3V1 Fbc V1 Section 3.1"
    }
  },

  // 2020201
  { FbcOnlyOneEachListOf, 
    "One of each list of allowed",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "There may be at most one instance of each of the "
    "following kinds of objects within a <model> object using Flux Balance "
    "Constraints: <listOfFluxBounds> and <listOfObjectives>.",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020202
  { FbcNoEmptyListOfs, 
    "ListOf elements cannot be empty",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The various ListOf subobjects with a <model> object are optional, "
    "but if present, these container object must not be empty. Specifically, "
    "if any of the following classes of objects are present on the <model>, "
    "it must not be empty: <listOfFluxBounds> and <listOfObjectives>.",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020203
  { FbcLOFluxBoundsAllowedElements, 
    "Allowed elements on ListOfFluxBounds",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "Apart from the general notes and annotation subobjects permitted on "
    "all SBML objects, a <listOfFluxBounds> container object may only "
    "contain <fluxBound> objects.",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020204
  { FbcLOObjectivesAllowedElements, 
    "Allowed elements on ListOfObjectives",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "Apart from the general notes and annotation subobjects permitted on "
    "all SBML objects, a <listOfObjectives> container object may only "
    "contain <objective> objects.",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020205
  { FbcLOFluxBoundsAllowedAttributes, 
    "Allowed attributes on ListOfFluxBounds",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <listOfFluxBounds> object may have the optional attributes 'metaid' "
    "and 'sboTerm' defined by SBML Level~3 Core. No other attributes from "
    "the SBML Level 3 Core namespace or the Flux Balance Constraints "
    "namespace are permitted on a <listOfFluxBounds> object. ",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020206
  { FbcLOObjectivesAllowedAttributes, 
    "Allowed attributes on ListOfObjectives",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <listOfFluxBounds> object may have the optional attributes 'metaid' "
    "and 'sboTerm' defined by SBML Level~3 Core. Additionally the "
    "<listOfObjectives> must contain the attribute 'activeObjective'. "
    "No other attributes from the SBML Level 3 Core namespace or the Flux "
    "Balance Constraints namespace are permitted on a "
    "<listOfFluxBounds> object. ",
    { "L3V1 Fbc V1 Section 3.3"
    }
  },

  // 2020207
  { FbcActiveObjectiveSyntax, 
    "Type of activeObjective attribute",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:activeObjective' on the <listOfObjectives> "
    "object must be of the data type SIdRef.",
    { "L3V1 Fbc V1 Section 3.2.2"
    }
  },

  // 2020208
  { FbcActiveObjectiveRefersObjective, 
    "ActiveObjective must reference Objective",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:activeObjective' on the <listOfObjectives> "
    "object must be the identifier of an existing <objective>.",
    { "L3V1 Fbc V1 Section 3.2.2"
    }
  },

  // 2020301
  { FbcSpeciesAllowedL3Attributes, 
    "Species allowed attributes",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A SBML <species> object may have the optional attributes 'fbc:charge' "
    "and 'fbc:chemicalFormula'. No other attributes from the Flux Balance "
    "Constraints namespaces are permitted on a <species>. ",
    { "L3V1 Fbc V1 Section 3.4"
    }
  },

  // 2020302
  { FbcSpeciesChargeMustBeInteger, 
    "Charge must be integer",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:charge' on SBML <species> object "
    "must be of the data type integer.",
    { "L3V1 Fbc V1 Section 3.4"
    }
  },

  // 2020303
  { FbcSpeciesFormulaMustBeString, 
    "Chemical formula must be string",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of attribute 'fbc:chemicalFormula' on the SBML <species> "
    "object must be set to a string consisting only of atomic names or user "
    "defined compounds and their occurrence.",
    { "L3V1 Fbc V1 Section 3.4"
    }
  },

  // 2020401
  { FbcFluxBoundAllowedL3Attributes, 
    "<fluxBound> may only have 'metaId' and 'sboTerm' from L3 namespace",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxBound> object may have the optional SBML Level 3 Core "
    "attributes 'metaid' and 'sboTerm'. No other attributes from the "
    "SBML Level 3 Core namespace are permitted on a <fluxBound>.",
    { "L3V1 Core, Section 3.2."
    }
  },
  
  // 2020402
  { FbcFluxBoundAllowedElements, 
    "<fluxBound> may only have <notes> and <annotations> from L3 Core",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxBound> object may have the optional SBML Level 3 Core "
    "subobjects for notes and annotations. No other elements from the "
    "SBML Level 3 Core namespace are permitted on a <fluxBound>.",
    { "L3V1 Core, Section 3.2."
    }
  },
  
  // 2020403
  { FbcFluxBoundRequiredAttributes, 
    "Invalid attribute found on <fluxBound> object",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxBound> object must have the required attributes 'fbc:reaction', "
    "'fbc:operation' and 'fbc:value', and may have the optional attributes "
    "'fbc:id' and 'fbc:name'. No other attributes from the SBML Level 3 Flux "
    "Balance Constraints namespace are permitted on a <fluxBound> object.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020404
  { FbcFluxBoundRectionMustBeSIdRef, 
    "Datatype for 'fbc:reaction' must be SIdRef",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:reaction' of a <fluxBound> must be of the data "
    "type SIdRef.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020405
  { FbcFluxBoundNameMustBeString, 
    "The attribute 'fbc:name' must be of the data type string",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:name' of a <fluxBound> must be of the data type "
    "string.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020406
  { FbcFluxBoundOperationMustBeEnum, 
    "The attribute 'fbc:operation' must be of data type FbcOperation",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:operation' of a <fluxBound> must be of the data "
    "type FbcOperation and thus it's value must be one of 'lessEqual', "
    "'greaterEqual' or 'equal'.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020407
  { FbcFluxBoundValueMustBeDouble, 
    "The attribute 'fbc:value' must be of the data type double",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:value' of a <fluxBound> must be of the "
    "data type double.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020408
  { FbcFluxBoundReactionMustExist, 
    "'fbc:reaction' must refer to valid reaction",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of the attribute 'fbc:reaction' of a <fluxBound> object "
    "must be the identifier of an existing <reaction> object defined in "
    "the enclosing <model> object.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020409
  { FbcFluxBoundsForReactionConflict, 
    "Conflicting set of FluxBounds for a reaction",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The combined set of all <fluxBound>'s with identical values for "
    "'fbc:reaction' must be consistent. That is while it is possible "
    "to define a lower and an upper bound for a reaction, it is not "
    "possible to define multiple lower or upper bounds.",
    { "L3V1 Fbc V1 Section 3.5"
    }
  },
  
  // 2020501
  { FbcObjectiveAllowedL3Attributes, 
    "<objective> may only have 'metaId' and 'sboTerm' from L3 namespace",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <objective> object may have the optional SBML Level 3 Core "
    "attributes 'metaid' and 'sboTerm'. No other attributes from the SBML "
    "Level 3 Core namespace are permitted on a <objective>. ",
    { "L3V1 Core, Section 3.2"
    }
  },
  
  // 2020502
  { FbcObjectiveAllowedElements, 
    "<objective> may only have <notes> and <annotations> from L3 Core",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <objective> object may have the optional SBML Level 3 Core subobjects "
    "for notes and annotations. No other elements from the SBML Level 3 Core "
    "namespace are permitted on a <objective>.",
    { "L3V1 Core, Section 3.2"
    }
  },
  
  // 2020503
  { FbcObjectiveRequiredAttributes, 
    "Invalid attribute found on <objective> object",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <objective> object must have the required attributes 'fbc:id' and "
    "'fbc:type' and may have the optional attribute 'fbc:name'. No other "
    "attributes from the SBML Level 3 Flux Balance Constraints namespace "
    "are permitted on a <objective> object.",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020504
  { FbcObjectiveNameMustBeString, 
    "The attribute 'fbc:name' must be of the data type string",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:name' on a <objective> must be of the data type "
    "'string'. ",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020505
  { FbcObjectiveTypeMustBeEnum, 
    "The attribute 'fbc:type' must be of data type FbcType.",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:type' on a <objective> must be of the data type "
    "FbcType. and thus its value must be "
    "one of 'minimize' or 'maximize.",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020506
  { FbcObjectiveOneListOfObjectives, 
    "An <objective> must have one <listOfFluxObjectives>.",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <objective> object must have one and only one instance of the "
    "<listOfFluxObjectives> object. ",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020507
  { FbcObjectiveLOFluxObjMustNotBeEmpty, 
    "<listOfFluxObjectives> subobject must not be empty",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The <listOfFluxObjectives> subobject within a <objective> object must "
    "not be empty.",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020508
  { FbcObjectiveLOFluxObjOnlyFluxObj, 
    "Invalid element found in <listOfFluxObjectives>",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "Apart from the general notes and annotation subobjects permitted on all "
    "SBML objects, a <listOfFluxObjectives> container object may only contain "
    "<fluxObjective> objects.",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020509
  { FbcObjectiveLOFluxObjAllowedAttribs, 
    "<listOfFluxObjectives> may only have 'metaId' and 'sboTerm' from L3 core",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <listOfFluxObjectives> object may have the optional 'metaid' and "
    "'sboTerm' defined by SBML Level 3 Core. No other attributes from the "
    "SBML Level 3 Core namespace or the Flux Balance Constraints namespace "
    "are permitted on a <listOfFluxObjectives> object",
    { "L3V1 Fbc V1, Section 3.6"
    }
  },
  
  // 2020601
  { FbcFluxObjectAllowedL3Attributes, 
    "<fluxObjective> may only have 'metaId' and 'sboTerm' from L3 namespace",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxObjective> object may have the optional SBML Level 3 Core "
    "attributes 'metaid' and 'sboTerm'. No other attributes from the SBML "
    "Level 3 Core namespace are permitted on a <fluxObjective>. ",
    { "L3V1 Core, Section 3.2"
    }
  },
  
  // 2020602
  { FbcFluxObjectAllowedElements, 
    "<fluxObjective> may only have <notes> and <annotations> from L3 Core",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxObjective> object may have the optional SBML Level 3 Core "
    "subobjects for notes and annotations. No other elements from the "
    "SBML Level 3 Core namespace are permitted on a <fluxObjective>.",
    { "L3V1 Core, Section 3.2"
    }
  },
  
  // 2020603
  { FbcFluxObjectRequiredAttributes, 
    "Invalid attribute found on <fluxObjective> object",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "A <fluxObjective> object must have the required attributes "
    "'fbc:reaction' and 'fbc:coefficient', and may have the optional "
    "attributes 'fbc:id' and 'fbc:name'. No other attributes from the "
    "SBML Level 3 Flux Balance Constraints namespace are permitted on "
    "a <fluxObjective> object.",
    { "L3V1 Fbc V1, Section 3.7"
    }
  },
  
  // 2020604
  { FbcFluxObjectNameMustBeString, 
    "The attribute 'fbc:name' must be of the data type string",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The attribute 'fbc:name' on a <fluxObjective> must be of the data "
    "type 'string'.",
    { "L3V1 Fbc V1, Section 3.7"
    }
  },
  
  // 2020605
  { FbcFluxObjectReactionMustBeSIdRef, 
    "Datatype for 'fbc:reaction' must be SIdRef",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of the attribute 'fbc:reaction' of a <fluxObjective> object "
    "must conform to the syntax of the SBML data type 'SIdRef'.",
    { "L3V1 Fbc V1, Section 3.7"
    }
  },
  
  // 2020606
  { FbcFluxObjectReactionMustExist, 
    "'fbc:reaction' must refer to valid reaction",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of the attribute 'fbc:reaction' of a "
    "<fluxObjective> object must be the identifier of an existing <reaction> "
    "object defined in the enclosing <model> object.",
    { "L3V1 Fbc V1, Section 3.7"
    }
  },
  
  // 2020607
  { FbcFluxObjectCoefficientMustBeDouble, 
    "The attribute 'fbc:coefficient' must be of the data type double",
    LIBSBML_CAT_GENERAL_CONSISTENCY, 
    LIBSBML_SEV_ERROR,
    "The value of the attribute 'fbc:coefficient' of a <fluxObjective> "
    "object must conform to the syntax of the SBML data type 'double'.",
    { "L3V1 Fbc V1, Section 3.7"
    }
  }

};

LIBSBML_CPP_NAMESPACE_END

/** @endcond */


/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    IdentifierConsistencyConstraints.cpp
 * @brief   IdentifierConsistency check constraints.  See SBML Wiki
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
 * Copyright 2011-2012 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#ifndef AddingConstraintsToValidator

#include <sbml/validator/VConstraint.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>
#include <sbml/packages/comp/util/SBMLUri.h>
#include <sbml/SBMLTypes.h>
#include <sbml/packages/comp/common/CompExtensionTypes.h>
#include <sbml/util/MetaIdFilter.h>
#include <sbml/util/IdFilter.h>

#include "ExtModelReferenceCycles.h"
#include "SubmodelReferenceCycles.h"
#include "UniquePortReferences.h"
#include "UniqueReplacedReferences.h"
#include "ClassReplacements.h"
#include "PackageIdReplacementCheck.h"

#endif

#include <sbml/common/libsbml-namespace.h>
#include <sbml/validator/ConstraintMacros.h>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE
/** @endcond */

class ReferencedModel
{
public:
  ReferencedModel(const Model & m, const Port & p)
  {
    referencedModel = 
      static_cast<const Model*>(p.getAncestorOfType(SBML_MODEL, "core"));
    if (referencedModel == NULL) 
    {
      referencedModel = static_cast<const Model*>
        (p.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
    }
  }

  ReferencedModel(const Model & m, const Deletion & d)
  {
    referencedModel = NULL;

    const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    if (sub != NULL)
    {
      std::string modelId = sub->getModelRef();

      const SBMLDocument * doc = d.getSBMLDocument();
      bool found = false;
      while (doc != NULL && found == false)
      {
        CompSBMLDocumentPlugin * docPlug = 
          (CompSBMLDocumentPlugin*)(doc->getPlugin("comp"));
      
        if (docPlug != NULL)
        {

          referencedModel = docPlug->getModelDefinition(modelId);
          if (referencedModel == NULL)
          {
            // may be an external model
            ExternalModelDefinition * emd = 
                              docPlug->getExternalModelDefinition(modelId);
            pre (emd != NULL);

            string locationURI = doc->getLocationURI();
            string uri = emd->getSource();

            const SBMLResolverRegistry& registry = 
                                 SBMLResolverRegistry::getInstance();
            doc = registry.resolve(uri, locationURI);
            if (doc != NULL)
            {
              if (emd->isSetModelRef() == false)
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else if (doc->getModel() != NULL &&
                doc->getModel()->isSetId() == true &&
                emd->getModelRef() == doc->getModel()->getId())
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else
              {
                modelId = emd->getModelRef();
              }
            }
          }
          else
          {
            found = true;
          }
        }
        else
        {
          found = true;
        }
      }
    }
  }

 
  
  ReferencedModel(const Model & m, const ReplacedElement & repE)
  {
    referencedModel = NULL;

    CompModelPlugin *plug = (CompModelPlugin*)(m.getPlugin("comp"));
    
    if ((plug != NULL) && (plug->getSubmodel(repE.getSubmodelRef()) != NULL))
    {
      std::string modelId = 
               (plug->getSubmodel(repE.getSubmodelRef()))->getModelRef();

      const SBMLDocument * doc = repE.getSBMLDocument();

      bool found = false;
      while (doc != NULL && found == false)
      {
        CompSBMLDocumentPlugin * docPlug = 
          (CompSBMLDocumentPlugin*)(doc->getPlugin("comp"));
      
        if (docPlug != NULL)
        {

          referencedModel = docPlug->getModelDefinition(modelId);
          if (referencedModel == NULL)
          {
            // may be an external model
            ExternalModelDefinition * emd = 
                              docPlug->getExternalModelDefinition(modelId);
            pre (emd != NULL);

            string locationURI = doc->getLocationURI();
            string uri = emd->getSource();

            const SBMLResolverRegistry& registry = 
                                 SBMLResolverRegistry::getInstance();
            doc = registry.resolve(uri, locationURI);
            if (doc != NULL)
            {
              if (emd->isSetModelRef() == false)
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else if (doc->getModel() != NULL &&
                doc->getModel()->isSetId() == true &&
                emd->getModelRef() == doc->getModel()->getId())
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else
              {
                modelId = emd->getModelRef();
              }
            }
          }
          else
          {
            found = true;
          }
        }
        else
        {
          found = true;
        }
      }
    }
  }

 
  
  ReferencedModel(const Model & m, const ReplacedBy & repBy)
  {
    referencedModel = NULL;

    CompModelPlugin *plug = (CompModelPlugin*)(m.getPlugin("comp"));
    
    if ((plug != NULL) && (plug->getSubmodel(repBy.getSubmodelRef()) != NULL))
    {
      std::string modelId = 
               (plug->getSubmodel(repBy.getSubmodelRef()))->getModelRef();

      const SBMLDocument * doc = repBy.getSBMLDocument();

      bool found = false;
      while (doc != NULL && found == false)
      {
        CompSBMLDocumentPlugin * docPlug = 
          (CompSBMLDocumentPlugin*)(doc->getPlugin("comp"));
      
        if (docPlug != NULL)
        {

          referencedModel = docPlug->getModelDefinition(modelId);
          if (referencedModel == NULL)
          {
            // may be an external model
            ExternalModelDefinition * emd = 
                              docPlug->getExternalModelDefinition(modelId);
            pre (emd != NULL);

            string locationURI = doc->getLocationURI();
            string uri = emd->getSource();

            const SBMLResolverRegistry& registry = 
                                 SBMLResolverRegistry::getInstance();
            doc = registry.resolve(uri, locationURI);
            if (doc != NULL)
            {
              if (emd->isSetModelRef() == false)
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else if (doc->getModel() != NULL &&
                doc->getModel()->isSetId() == true &&
                emd->getModelRef() == doc->getModel()->getId())
              {
                referencedModel = doc->getModel();
                found = true;
              }
              else
              {
                modelId = emd->getModelRef();
              }
            }
          }
          else
          {
            found = true;
          }
        }
        else
        {
          found = true;
        }
      }
    }
  }

 
  
  ReferencedModel(const Model & m, const SBaseRef & sbRef)
  {
    referencedModel = NULL;
    
    if (sbRef.getParentSBMLObject() != NULL)
    {
      int tc = sbRef.getParentSBMLObject()->getTypeCode();
      //const SBMLDocument * doc = sbRef.getSBMLDocument();
      
      ReferencedModel *ref;
      std::string idRef;
      std::string metaIdRef;
      std::string modelId;
      const Model* parentRefModel = NULL;
      const ReplacedElement * repE = NULL;
      const ReplacedBy* repBy = NULL;
      const Deletion * del = NULL;
      const Port * port = NULL;
      const SBaseRef * parent = NULL;
      const SBaseRef * grandParent = NULL;
      int tc1;


      switch (tc)
      {
        case SBML_COMP_REPLACEDELEMENT:
          repE = 
            static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject());
          ref = new ReferencedModel(m, *(repE));
          parentRefModel = ref->getReferencedModel();
          idRef = repE->getIdRef();
          metaIdRef = repE->getMetaIdRef();
          break;
        case SBML_COMP_REPLACEDBY:
          repBy = 
            static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject());
          ref = new ReferencedModel(m, *(repBy));
          parentRefModel = ref->getReferencedModel();
          idRef = repBy->getIdRef();
          metaIdRef = repBy->getMetaIdRef();
          break;
        case SBML_COMP_PORT:
          port = 
            static_cast<const Port*>(sbRef.getParentSBMLObject());
          ref = new ReferencedModel(m, *(port));
          parentRefModel = ref->getReferencedModel();
          idRef = port->getIdRef();
          metaIdRef = port->getMetaIdRef();
         break;
        case SBML_COMP_DELETION:
          del = 
            static_cast<const Deletion*>(sbRef.getParentSBMLObject());
          ref = new ReferencedModel(m, *(del));
          parentRefModel = ref->getReferencedModel();
          idRef = del->getIdRef();
          metaIdRef = del->getMetaIdRef();
         break;
        case SBML_COMP_SBASEREF:
          parent = static_cast<const SBaseRef*>(sbRef.getParentSBMLObject());
          idRef = parent->getIdRef();
          metaIdRef = parent->getMetaIdRef();
          if (idRef.empty() == false)
          {
            mReferences.push_back(make_pair(idRef, "id"));
          }
          else
          {
            mReferences.push_back(make_pair(metaIdRef, "metaid"));
          }
          grandParent = static_cast<const SBaseRef*>
                                   (parent->getParentSBMLObject());
          tc1 = grandParent->getTypeCode();
          while (tc1 == SBML_COMP_SBASEREF)
          {
            idRef = grandParent->getIdRef();
            metaIdRef = grandParent->getMetaIdRef();
            if (idRef.empty() == false)
            {
              mReferences.push_back(make_pair(idRef, "id"));
            }
            else
            {
              mReferences.push_back(make_pair(metaIdRef, "metaid"));
            }
            grandParent = static_cast<const SBaseRef*>
                                      (grandParent->getParentSBMLObject());
            tc1 = grandParent->getTypeCode();
          }
          switch (tc1)
          {
            case SBML_COMP_REPLACEDELEMENT:
              repE = 
                static_cast<const ReplacedElement*>(grandParent);
              ref = new ReferencedModel(m, *(repE));
              parentRefModel = ref->getReferencedModel();
              idRef = repE->getIdRef();
              metaIdRef = repE->getMetaIdRef();
              break;
            case SBML_COMP_REPLACEDBY:
              repBy = 
                static_cast<const ReplacedBy*>(grandParent);
              ref = new ReferencedModel(m, *(repBy));
              parentRefModel = ref->getReferencedModel();
              idRef = repBy->getIdRef();
              metaIdRef = repBy->getMetaIdRef();
              break;
            case SBML_COMP_PORT:
              port = 
                static_cast<const Port*>(grandParent);
              ref = new ReferencedModel(m, *(port));
              parentRefModel = ref->getReferencedModel();
              idRef = port->getIdRef();
              metaIdRef = port->getMetaIdRef();
             break;
            case SBML_COMP_DELETION:
              del = 
                static_cast<const Deletion*>(grandParent);
              ref = new ReferencedModel(m, *(del));
              parentRefModel = ref->getReferencedModel();
              idRef = del->getIdRef();
              metaIdRef = del->getMetaIdRef();
             break;
          }
        break;
      }
      
      if (parentRefModel != NULL)
      {
        const SBMLDocument* doc = parentRefModel->getSBMLDocument();
        CompSBMLDocumentPlugin* docPlug = (CompSBMLDocumentPlugin*)(doc->getPlugin("comp"));
        CompModelPlugin *plug1 = 
                        (CompModelPlugin*)(parentRefModel->getPlugin("comp"));
        
        if (docPlug != NULL && plug1 != NULL)
        {
          if (idRef.empty() == false)
          {
            pre (plug1->getSubmodel(idRef) != NULL);

            modelId = (plug1->getSubmodel(idRef))->getModelRef();
          }
          else
          {
            for (unsigned int i = 0; i < plug1->getNumSubmodels(); i++)
            {
              if (plug1->getSubmodel(i)->getMetaId() == metaIdRef)
              {
                modelId = plug1->getSubmodel(i)->getModelRef();
                break;
              }
            }
          }

          referencedModel = docPlug->getModelDefinition(modelId);
          if (referencedModel == NULL)
          {
            /* may be an external model */
            ExternalModelDefinition * emd = 
                                docPlug->getExternalModelDefinition(modelId);
            pre (emd != NULL);

            string locationURI = doc->getLocationURI();
            string uri = emd->getSource();

            const SBMLResolverRegistry& registry = 
                                  SBMLResolverRegistry::getInstance();
            SBMLDocument *newDoc = registry.resolve(uri, locationURI);
            pre(newDoc != NULL);
            referencedModel = newDoc->getModel();
          }

          while (mReferences.empty() == false)
          {
            size_t numRefs = mReferences.size();
            if (mReferences.at(numRefs - 1).second == "id")
            {
              idRef = mReferences.at(numRefs -1 ).first;
              metaIdRef = "";
            }
            else
            {
              metaIdRef = mReferences.at(numRefs -1 ).first;
              idRef = "";
            }
            CompModelPlugin *plug1 = 
                        (CompModelPlugin*)(referencedModel->getPlugin("comp"));
            
            if (docPlug != NULL && plug1 != NULL)
            {
              if (idRef.empty() == false)
              {
                pre (plug1->getSubmodel(idRef) != NULL);

                modelId = (plug1->getSubmodel(idRef))->getModelRef();
              }
              else
              {
                for (unsigned int i = 0; i < plug1->getNumSubmodels(); i++)
                {
                  if (plug1->getSubmodel(i)->getMetaId() == metaIdRef)
                  {
                    modelId = plug1->getSubmodel(i)->getModelRef();
                    break;
                  }
                }
              }

              referencedModel = docPlug->getModelDefinition(modelId);
              if (referencedModel == NULL)
              {
                /* may be an external model */
                ExternalModelDefinition * emd = 
                                    docPlug->getExternalModelDefinition(modelId);
                pre (emd != NULL);

                string locationURI = doc->getLocationURI();
                string uri = emd->getSource();

                const SBMLResolverRegistry& registry = 
                                      SBMLResolverRegistry::getInstance();
                SBMLDocument *newDoc = registry.resolve(uri, locationURI);
                pre(newDoc != NULL);
                referencedModel = newDoc->getModel();
              }
            }
            mReferences.erase(mReferences.end()-1);
          }
        }
      }
    }
  }


  const Model * getReferencedModel()
  {
    return referencedModel;
  }

private:

  const Model* referencedModel;
  vector< pair< std::string, std::string > >  mReferences;
};

//*************************************

//SBase  constraints

// 20101 - caught at read
// 20102 - caught at read
// 20103 - caught at read
// 20104 - caught at read
// 20105 - caught at read

//*************************************

//SBML class  constraints

// 20201 - caught at read
// 20202 - caught at read
// 20203 - caught by checkConsistency
// 20204 - caught by checkConsistency
// 20205 - caught at read
// 20206 - caught at read
// 20207 - caught at read
// 20208 - caught at read
// 20209 - caught at read
// 20210 - caught at read
// 20211 - caught at read

//*************************************

//ExternalModelDefinition  constraints

// 20301 - caught at read
// 20302 - caught at read
// 20303 - caught at read

//20304
START_CONSTRAINT (CompReferenceMustBeL3, ExternalModelDefinition, emd)
{
  pre (emd.isSetSource() == true);
  pre (emd.isSetId() == true);
 
  bool fail = false;

  msg = "<externalModelDefinition> '";
  msg += emd.getId();
  msg += "' refers to a URI '";
  msg += emd.getSource();
  msg += "' which is not an SBML Level 3 document.";

  const SBMLResolverRegistry& registry = SBMLResolverRegistry::getInstance();
  const SBMLDocument* doc = emd.getSBMLDocument();
  pre(doc != NULL);
  string locationURI = doc->getLocationURI();
  string uri = emd.getSource();
  doc = NULL;

  doc = registry.resolve(uri, locationURI);
  pre (doc != NULL);

  if (doc->getLevel() != 3) 
  {
    fail = true;
  }

  delete doc;
  
  inv(fail == false);
}
END_CONSTRAINT

//20305
START_CONSTRAINT (CompModReferenceMustIdOfModel, ExternalModelDefinition, emd)
{
  pre (emd.isSetSource() == true);
  pre (emd.isSetId() == true);
  pre (emd.isSetModelRef() == true);
  
  bool fail = false;

  msg = "<externalModelDefinition> '";
  msg += emd.getId() ;
  msg += "' refers to a model with id '";
  msg += emd.getModelRef();
  msg += "' that does not exist in the referenced document.";

  const SBMLResolverRegistry& registry = SBMLResolverRegistry::getInstance();
  const SBMLDocument* doc = emd.getSBMLDocument();
  pre(doc != NULL);
  string locationURI = doc->getLocationURI();
  string uri = emd.getSource();

  //SBMLUri* resolved = registry.resolveUri(uri, locationURI);
  //pre(resolved != NULL )
  //string resolvedURI = resolved->getUri();
  //delete resolved;
  doc = registry.resolve(uri, locationURI);
  pre(doc != NULL);
  pre(doc->getLevel() == 3);

  const CompSBMLDocumentPlugin* csdp = 
    static_cast<const CompSBMLDocumentPlugin*>(doc->getPlugin("comp"));
  if (csdp == NULL) 
  {
    const Model* model = doc->getModel();
    if (model==NULL || (model->getId() != emd.getModelRef())) {
      fail = true;
    }
  }
  else 
  {
    const SBase* referencedmod = csdp->getModel(emd.getModelRef());
    if (referencedmod == NULL) 
    {
      fail = true;
    }
  }

  delete doc;
  inv(fail == false);
}
END_CONSTRAINT

//TODO: 20306 - caught at read md5
// 20307 - caught at read anyURI
// 20308 - caught at read
// 20309 - string
// 20310 
EXTERN_CONSTRAINT( CompCircularExternalModelReference, ExtModelReferenceCycles)

//90101
START_CONSTRAINT (CompUnresolvedReference, ExternalModelDefinition, emd)
{
  pre (emd.isSetSource() == true);
  
  const SBMLDocument* doc = emd.getSBMLDocument();
  pre(doc != NULL);
  string locationURI = doc->getLocationURI();
  string uri = emd.getSource();

  const SBMLResolverRegistry& registry = SBMLResolverRegistry::getInstance();
  SBMLUri* resolved = registry.resolveUri(uri, locationURI);

  bool fail = false;

  msg = "<externalModelDefinition> '";
  msg += emd.getId() ;
  msg += "' refers to a source '";
  msg += emd.getSource();
  msg += "' that cannot be accessed from here. Further checks relating to";
  msg += " this document cannot be performed.";

  if (resolved == NULL) 
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT


//*************************************

// 204xx - not used 

//*************************************

//Model  constraints

//20501 - caught at read
//20502 - caught at read
//20503 - caught at read
//20504 - caught at read
//20505 - caught at read
//20506 - caught at read


//*************************************

//Submodel constraints

//20601 - caught at read
//20602 - caught at read
//20603 - caught at read
//20604 - caught at read
//20605 - caught at read
//20606 - caught at read
//20607 - caught at read
//20608 - caught at read

//20609-20612 - non existant

//20613 - caught at read
//20614 - caught at read

//20615
START_CONSTRAINT (CompSubmodelMustReferenceModel, Submodel, s)
{
  pre (s.isSetModelRef() == true);

  bool fail = true;

  msg = "<submodel> '";
  msg += s.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                     (s.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                      (s.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to a model with id '";
  msg += s.getModelRef();
  msg += "' that does not exist in the referenced document.";

  // do we reference the actual model
  // do not report this here as it is another error
  if (s.getModelRef() == m.getId())
  {
    fail = false;
  }

  if (fail == true)
  {
    // do we refernce an external modelDefinition
    CompSBMLDocumentPlugin *docPlug = (CompSBMLDocumentPlugin*)
      (m.getSBMLDocument()->getPlugin("comp"));
    pre (docPlug != NULL);

    ModelDefinition * md = docPlug->getModelDefinition(s.getModelRef());
    if (md == NULL)
    {
      ExternalModelDefinition * ext = 
        docPlug->getExternalModelDefinition(s.getModelRef());

      if (ext != NULL)
      {
        fail = false;
      }
    }
    else
    {
      fail = false;
    }
  }

  inv(fail == false);
}
END_CONSTRAINT

//20616
START_CONSTRAINT (CompSubmodelCannotReferenceSelf, Submodel, s)
{
  pre (s.isSetModelRef() == true);

  bool fail = false;

  msg = "<submodel> '";
  msg += s.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                (s.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) 
  {
    mod = static_cast<const Model*>
                      (s.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to the enclosing model with id '";
  msg += s.getModelRef();
  msg += "'.";

  if (m.getId() == s.getModelRef())
  {
    fail = true;
  }
  
  inv(fail == false);

}
END_CONSTRAINT

// 20617 
EXTERN_CONSTRAINT( CompModCannotCircularlyReferenceSelf, 
                                                SubmodelReferenceCycles)

// 20618 - 20621 non existant

//20622
START_CONSTRAINT (CompTimeConversionMustBeParameter, Submodel, s)
{
  pre (s.isSetTimeConversionFactor() == true);

  bool fail = false;

  msg = "The 'timeConversionFactor' of <submodel> '";
  msg += s.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                (s.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (s.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " is set to '";
  msg += s.getTimeConversionFactor();
  msg += "' which is not a <parameter> within the <model>.";

  if (m.getParameter(s.getTimeConversionFactor()) == NULL)
  {
    fail = true;
  }
  
  inv(fail == false);

}
END_CONSTRAINT

//20623
START_CONSTRAINT (CompExtentConversionMustBeParameter, Submodel, s)
{
  pre (s.isSetExtentConversionFactor() == true);

  bool fail = false;

  msg = "The 'extentConversionFactor' of <submodel> '";
  msg += s.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                (s.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                      (s.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " is set to '";
  msg += s.getExtentConversionFactor();
  msg += "' which is not a <parameter> within the <model>.";

  if (m.getParameter(s.getExtentConversionFactor()) == NULL)
  {
    fail = true;
  }
  
  inv(fail == false);

}
END_CONSTRAINT

//*************************************

//SBaseRef constraints
// -  need to implement for each object that derives from SBaseRef
// Port; Deletion; ReplacedElement; ReplacedBy

//20701
// Port doesnt have portRef

// 20701 - deletion
START_CONSTRAINT (CompPortRefMustReferencePort, Deletion, d)
{
  pre(d.isSetPortRef());
  
  bool fail = false;

  const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'portRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getPortRef();
  msg += "' which is not a <port> within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'.";

  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  CompModelPlugin *plug1 = 
                  (CompModelPlugin*)(referencedModel->getPlugin("comp"));
  pre (plug1 != NULL);

  if (plug1->getPort(d.getPortRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20701 - replacedElement
START_CONSTRAINT (CompPortRefMustReferencePort, ReplacedElement, repE)
{
  pre(repE.isSetPortRef());
  pre(repE.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'portRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getPortRef();
  msg += "' which is not a <port> within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repE);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  CompModelPlugin *plug1 = 
                   (CompModelPlugin*)(referencedModel->getPlugin("comp"));
  pre (plug1 != NULL);

  if (plug1->getPort(repE.getPortRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20701 - replacedBy
START_CONSTRAINT (CompPortRefMustReferencePort, ReplacedBy, repBy)
{
  pre(repBy.isSetPortRef());
  pre(repBy.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'portRef' of a <replacedBy>";
  msg += " is set to '";
  msg += repBy.getPortRef();
  msg += "' which is not a <port> within the <model> referenced by ";
  msg += "submodel '";
  msg += repBy.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repBy);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  CompModelPlugin *plug1 = 
                  (CompModelPlugin*)(referencedModel->getPlugin("comp"));
  pre (plug1 != NULL);

  if (plug1->getPort(repBy.getPortRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20701 - sBaseRef
START_CONSTRAINT (CompPortRefMustReferencePort, SBaseRef, sbRef)
{
  pre(sbRef.isSetPortRef());

  bool fail = false;

  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'portRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getPortRef();
  msg += "' which is not a <port> within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  CompModelPlugin *plug1 = 
                  (CompModelPlugin*)(referencedModel->getPlugin("comp"));
  pre (plug1 != NULL);

  if (plug1->getPort(sbRef.getPortRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20702
//20702 - port
START_CONSTRAINT (CompIdRefMustReferenceObject, Port, p)
{
  pre(p.isSetIdRef());
  
  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  msg = "The 'idRef' of a <port>";
  msg += " is set to '";
  msg += p.getIdRef();
  msg += "' which is not an element within the <model>.";

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  ReferencedModel *ref = new ReferencedModel(m, p);
  const Model* mod = ref->getReferencedModel();
  
  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  if (mIds.contains(p.getIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20702 - deletion
START_CONSTRAINT (CompIdRefMustReferenceObject, Deletion, d)
{
  pre(d.isSetIdRef());

  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);
  
  bool fail = false;

  const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'idRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'.";

  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                                (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  if (mIds.contains(d.getIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20702 - replacedElement
START_CONSTRAINT (CompIdRefMustReferenceObject, ReplacedElement, repE)
{
  pre(repE.isSetIdRef());
  pre(repE.isSetSubmodelRef());

  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  msg = "The 'idRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel ref(m, repE);
  const Model* referencedModel = ref.getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                               (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  if (mIds.contains(repE.getIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20702 - replacedBy
START_CONSTRAINT (CompIdRefMustReferenceObject, ReplacedBy, repBy)
{
  pre(repBy.isSetIdRef());
  pre(repBy.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'idRef' of a <replacedBy>";
  msg += " is set to '";
  msg += repBy.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repBy.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repBy);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                               (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  if (mIds.contains(repBy.getIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20702 - sBaseRef
START_CONSTRAINT (CompIdRefMustReferenceObject, SBaseRef, sbRef)
{
  pre(sbRef.isSetIdRef());

  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'idRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }
  
  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                                (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;


  if (mIds.contains(sbRef.getIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20703
//20703 - port
START_CONSTRAINT (CompUnitRefMustReferenceUnitDef, Port, p)
{
  pre(p.isSetUnitRef());
  
  bool fail = false;

  msg = "The 'unitRef' of a <port>";
  msg += " is set to '";
  msg += p.getUnitRef();
  msg += "' which is not a <unitDefinition> within the <model>.";

  if (m.getUnitDefinition(p.getUnitRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20703 - deletion
START_CONSTRAINT (CompUnitRefMustReferenceUnitDef, Deletion, d)
{
  pre(d.isSetUnitRef());
  
  bool fail = false;

  const Submodel * sub = static_cast<const Submodel*>
                         (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'unitRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getUnitRef();
  msg += "' which is not a <unitDefinition> within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'.";

  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  if (referencedModel->getUnitDefinition(d.getUnitRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20703 - replacedElement
START_CONSTRAINT (CompUnitRefMustReferenceUnitDef, ReplacedElement, repE)
{
  pre(repE.isSetUnitRef());
  pre(repE.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'unitRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getUnitRef();
  msg += "' which is not a <unitDefinition> within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repE);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  if (referencedModel->getUnitDefinition(repE.getUnitRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20703 - replacedBy
START_CONSTRAINT (CompUnitRefMustReferenceUnitDef, ReplacedBy, repBy)
{
  pre(repBy.isSetUnitRef());
  pre(repBy.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'unitRef' of a <replacedBy>";
  msg += " is set to '";
  msg += repBy.getUnitRef();
  msg += "' which is not a <unitDefinition> within the <model> referenced by ";
  msg += "submodel '";
  msg += repBy.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repBy);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  if (referencedModel->getUnitDefinition(repBy.getUnitRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20703 - sBaseRef
START_CONSTRAINT (CompUnitRefMustReferenceUnitDef, SBaseRef, sbRef)
{
  pre(sbRef.isSetUnitRef());

  bool fail = false;

  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'unitRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getUnitRef();
  msg += "' which is not a <unitDefinition> within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  if (referencedModel->getUnitDefinition(sbRef.getUnitRef()) == NULL)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }


  inv(fail == false);
}
END_CONSTRAINT

// 20704
//20704 - port
START_CONSTRAINT (CompMetaIdRefMustReferenceObject, Port, p)
{
  pre(p.isSetMetaIdRef());
  
  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  msg = "The 'metaIdRef' of a <port>";
  msg += " is set to '";
  msg += p.getMetaIdRef();
  msg += "' which is not an element within the <model>.";

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  ReferencedModel *ref = new ReferencedModel(m, p);
  const Model* mod = ref->getReferencedModel();

  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }


  if (mIds.contains(p.getMetaIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20704 - deletion
START_CONSTRAINT (CompMetaIdRefMustReferenceObject, Deletion, d)
{
  pre(d.isSetMetaIdRef());
  
  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'metaIdRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'.";

  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                                (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  if (mIds.contains(d.getMetaIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20704 - replacedElement
START_CONSTRAINT (CompMetaIdRefMustReferenceObject, ReplacedElement, repE)
{
  pre(repE.isSetMetaIdRef());
  pre(repE.isSetSubmodelRef());

  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  msg = "The 'metaidRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repE);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                               (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  if (mIds.contains(repE.getMetaIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20704 - replacedBy
START_CONSTRAINT (CompMetaIdRefMustReferenceObject, ReplacedBy, repBy)
{
  pre(repBy.isSetMetaIdRef());
  pre(repBy.isSetSubmodelRef());

  bool fail = false;

  msg = "The 'metaIdRef' of a <replacedBy>";
  msg += " is set to '";
  msg += repBy.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repBy.getSubmodelRef();
  msg += "'.";

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, repBy);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                                (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  if (mIds.contains(repBy.getMetaIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20704 - sBaseRef
START_CONSTRAINT (CompMetaIdRefMustReferenceObject, SBaseRef, sbRef)
{
  pre(sbRef.isSetMetaIdRef());

  /* only log this if there are no unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == false);

  bool fail = false;

  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'metaIdRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }

  /* need to be using the correct model */
  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* referencedModel = ref->getReferencedModel();

  pre (referencedModel != NULL);

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  List* allElements = const_cast<Model*>
                                (referencedModel)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  if (mIds.contains(sbRef.getMetaIdRef()) == false)
  {
    /* take out for now since I was right the first time
     * the reference should be there without need to instantiate
     */
    //// it is possible that the referenced model needs to actually instantiate
    //// its submodels to find the reference
    //// we are not going to do that here so if there are submodels
    //// give it the benefit of the doubt and do not report the id as missing
    //const CompModelPlugin * plug = static_cast<const CompModelPlugin*>
    //                                   (referencedModel->getPlugin("comp"));
    //if (plug == NULL || plug->getNumSubmodels() == 0)
    //{
    //  fail = true;
    //}
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT


// 20705
// 20705 - port
START_CONSTRAINT (CompParentOfSBRefChildMustBeSubmodel, Port, port)
{
  pre (port.isSetSBaseRef());

  bool fail = false;

  if (port.isSetIdRef() == true || port.isSetMetaIdRef() == true)
  {
    if (port.isSetIdRef() == true)
    {
      msg = "The 'idRef' of a <replacedElement>";
      msg += " is set to '";
      msg += port.getIdRef();
    }
    else
    {
      msg = "The 'metaIdRef' of a <replacedElement>";
      msg += " is set to '";
      msg += port.getMetaIdRef();
    }
    msg += "' which is not a submodel within the <model>.";

    /* need to be using the correct model */
    ReferencedModel *ref = new ReferencedModel(m, port);
    const Model* mod = ref->getReferencedModel();

    pre (mod != NULL);

    CompModelPlugin *plug = (CompModelPlugin*)(mod->getPlugin("comp"));
    
    pre (plug != NULL);

    if (port.isSetIdRef() == true)
    {
      if (plug->getSubmodel(port.getIdRef()) == NULL)
      {
        fail = true;
      }
    }
    else
    {
      // must be a metaidref
      std::string ref = port.getMetaIdRef();
      bool found = false;
      unsigned int i = 0;
      while (found == false &&  i < plug->getNumSubmodels())
      {
        if (ref == plug->getSubmodel(i)->getMetaId())
        {
          found = true;
        }

        i++;
      }
      if (found == false)
      {
        fail = true;
      }
    }
  }
  else
  {
    fail = true;

    if (port.isSetUnitRef() == true)
    {
      msg = "The 'unitRef' of a <replacedElement>";
      msg += " is set to '";
      msg += port.getUnitRef();
    }
    msg += "' which is not a submodel within the <model>.";
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20705 - deletion
START_CONSTRAINT (CompParentOfSBRefChildMustBeSubmodel, Deletion, del)
{
  pre (del.isSetSBaseRef());

  bool fail = false;

  const Submodel * sub = static_cast<const Submodel*>
                         (del.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  if (del.isSetIdRef() == true 
    || del.isSetMetaIdRef() == true
    || del.isSetPortRef() == true)
  {
    if (del.isSetIdRef() == true)
    {
      msg = "The 'idRef' of a <deletion>";
      msg += " is set to '";
      msg += del.getIdRef();
    }
    else if (del.isSetPortRef() == true)
    {
      msg = "The 'portRef' of a <deletion>";
      msg += " is set to '";
      msg += del.getPortRef();
    }
    else
    {
      msg = "The 'metaIdRef' of a <deletion>";
      msg += " is set to '";
      msg += del.getMetaIdRef();
    }
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += sub->getId();
    msg += "'.";

    /* need to be using the correct model */
    ReferencedModel *ref = new ReferencedModel(m, del);
    const Model* referencedModel = ref->getReferencedModel();

    pre (referencedModel != NULL);

    CompModelPlugin *plug1 = 
                    (CompModelPlugin*)(referencedModel->getPlugin("comp"));
    pre (plug1 != NULL);

    if (del.isSetIdRef() == true)
    {
      if (plug1->getSubmodel(del.getIdRef()) == NULL)
      {
        fail = true;
      }
    }
    else if (del.isSetPortRef() == true)
    {
      bool found = false;
      Port* port = plug1->getPort(del.getPortRef());
      if (port->isSetIdRef() == true)
      {
        if (plug1->getSubmodel(port->getIdRef()) != NULL)
        {
          found = true;
        }
      }
      else if (port->isSetMetaIdRef() == true)
      {
        unsigned int i = 0;
        while (found == false &&  i < plug1->getNumSubmodels())
        {
          if (port->getMetaIdRef() == plug1->getSubmodel(i)->getMetaId())
          {
            found = true;
          }

          i++;
        }
      }

      if (found == false)
      {
        fail = true;
      }
    }
    else
    {
      // must be a metaidref
      std::string ref = del.getMetaIdRef();
      bool found = false;
      unsigned int i = 0;
      while (found == false &&  i < plug1->getNumSubmodels())
      {
        if (ref == plug1->getSubmodel(i)->getMetaId())
        {
          found = true;
        }

        i++;
      }
      if (found == false)
      {
        fail = true;
      }
    }
  }
  else
  {
    fail = true;

    msg = "The 'unitRef' of a <deletion>";
    msg += " is set to '";
    msg += del.getUnitRef();
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += sub->getId();
    msg += "'.";  
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20705 - replacedElement
START_CONSTRAINT (CompParentOfSBRefChildMustBeSubmodel, ReplacedElement, repE)
{
  pre (repE.isSetSBaseRef());

  bool fail = false;

  if (repE.isSetIdRef() == true 
    || repE.isSetMetaIdRef() == true
    || repE.isSetPortRef() == true)
  {
    if (repE.isSetIdRef() == true)
    {
      msg = "The 'idRef' of a <replacedElement>";
      msg += " is set to '";
      msg += repE.getIdRef();
    }
    else if (repE.isSetMetaIdRef() == true)
    {
      msg = "The 'metaIdRef' of a <replacedElement>";
      msg += " is set to '";
      msg += repE.getMetaIdRef();
    }
    else
    {
      msg = "The 'portRef' of a <replacedElement>";
      msg += " is set to '";
      msg += repE.getPortRef();
    }
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += repE.getSubmodelRef();
    msg += "'.";

    /* need to be using the correct model */
    ReferencedModel *ref = new ReferencedModel(m, repE);
    const Model* referencedModel = ref->getReferencedModel();

    pre (referencedModel != NULL);

    CompModelPlugin *plug1 = 
                    (CompModelPlugin*)(referencedModel->getPlugin("comp"));
    pre (plug1 != NULL);

    if (repE.isSetIdRef() == true)
    {
      if (plug1->getSubmodel(repE.getIdRef()) == NULL)
      {
        fail = true;
      }
    }
    else if (repE.isSetPortRef() == true)
    {
      bool found = false;
      Port* port = plug1->getPort(repE.getPortRef());
      if (port->isSetIdRef() == true)
      {
        if (plug1->getSubmodel(port->getIdRef()) != NULL)
        {
          found = true;
        }
      }
      else if (port->isSetMetaIdRef() == true)
      {
        unsigned int i = 0;
        while (found == false &&  i < plug1->getNumSubmodels())
        {
          if (port->getMetaIdRef() == plug1->getSubmodel(i)->getMetaId())
          {
            found = true;
          }

          i++;
        }
      }

      if (found == false)
      {
        fail = true;
      }
    }
    else
    {
      // must be a metaidref
      std::string ref = repE.getMetaIdRef();
      bool found = false;
      unsigned int i = 0;
      while (found == false &&  i < plug1->getNumSubmodels())
      {
        if (ref == plug1->getSubmodel(i)->getMetaId())
        {
          found = true;
        }

        i++;
      }
      if (found == false)
      {
        fail = true;
      }
    }
  }
  else
  {
    fail = true;

    msg = "The 'unitRef' of a <replacedElement>";
    msg += " is set to '";
    msg += repE.getUnitRef();
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += repE.getSubmodelRef();
    msg += "'.";  
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20705 - replacedBy
START_CONSTRAINT (CompParentOfSBRefChildMustBeSubmodel, ReplacedBy, repE)
{
  pre (repE.isSetSBaseRef());

  bool fail = false;

  if (repE.isSetIdRef() == true 
    || repE.isSetMetaIdRef() == true
    || repE.isSetPortRef() == true)
  {
    if (repE.isSetIdRef() == true)
    {
      msg = "The 'idRef' of a <replacedBy>";
      msg += " is set to '";
      msg += repE.getIdRef();
    }
    else if (repE.isSetMetaIdRef() == true)
    {
      msg = "The 'metaIdRef' of a <replacedBy>";
      msg += " is set to '";
      msg += repE.getMetaIdRef();
    }
    else
    {
      msg = "The 'portRef' of a <replacedBy>";
      msg += " is set to '";
      msg += repE.getPortRef();
    }
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += repE.getSubmodelRef();
    msg += "'.";

    /* need to be using the correct model */
    ReferencedModel *ref = new ReferencedModel(m, repE);
    const Model* referencedModel = ref->getReferencedModel();

    pre (referencedModel != NULL);

    CompModelPlugin *plug1 = 
                    (CompModelPlugin*)(referencedModel->getPlugin("comp"));
    pre (plug1 != NULL);

    if (repE.isSetIdRef() == true)
    {
      if (plug1->getSubmodel(repE.getIdRef()) == NULL)
      {
        fail = true;
      }
    }
    else if (repE.isSetPortRef() == true)
    {
      bool found = false;
      Port* port = plug1->getPort(repE.getPortRef());
      if (port->isSetIdRef() == true)
      {
        if (plug1->getSubmodel(port->getIdRef()) != NULL)
        {
          found = true;
        }
      }
      else if (port->isSetMetaIdRef() == true)
      {
        unsigned int i = 0;
        while (found == false &&  i < plug1->getNumSubmodels())
        {
          if (port->getMetaIdRef() == plug1->getSubmodel(i)->getMetaId())
          {
            found = true;
          }

          i++;
        }
      }

      if (found == false)
      {
        fail = true;
      }
    }
    else
    {
      // must be a metaidref
      std::string ref = repE.getMetaIdRef();
      bool found = false;
      unsigned int i = 0;
      while (found == false &&  i < plug1->getNumSubmodels())
      {
        if (ref == plug1->getSubmodel(i)->getMetaId())
        {
          found = true;
        }

        i++;
      }
      if (found == false)
      {
        fail = true;
      }
    }
  }
  else
  {
    fail = true;

    msg = "The 'unitRef' of a <replacedBy>";
    msg += " is set to '";
    msg += repE.getUnitRef();
    msg += "' which is not a submodel within the <model> referenced by ";
    msg += "submodel '";
    msg += repE.getSubmodelRef();
    msg += "'.";  
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20705 - sBaseRef
START_CONSTRAINT (CompParentOfSBRefChildMustBeSubmodel, SBaseRef, sbRef)
{
  pre (sbRef.isSetSBaseRef());

  bool fail = false;

  if (sbRef.isSetIdRef() == true 
    || sbRef.isSetMetaIdRef() == true
    || sbRef.isSetPortRef() == true)
  {
    if (sbRef.isSetIdRef() == true)
    {
      msg = "The 'idRef' of a <sBaseRef>";
      msg += " is set to '";
      msg += sbRef.getIdRef();
    }
    else if (sbRef.isSetPortRef() == true)
    {
      msg = "The 'portRef' of a <sBaseRef>";
      msg += " is set to '";
      msg += sbRef.getPortRef();
    }
    else
    {
      msg = "The 'metaIdRef' of a <sbaseRef>";
      msg += " is set to '";
      msg += sbRef.getMetaIdRef();
    }
    msg += "' which is not a submodel within the referenced <model>.";

    /* need to be using the correct model */
    /* need to be using the correct model */
    ReferencedModel *ref = new ReferencedModel(m, sbRef);
    const Model* referencedModel = ref->getReferencedModel();

    pre (referencedModel != NULL);

    CompModelPlugin *plug = (CompModelPlugin*)
                            (referencedModel->getPlugin("comp"));
    
    pre (plug != NULL);

    if (sbRef.isSetIdRef() == true)
    {
      if (plug->getSubmodel(sbRef.getIdRef()) == NULL)
      {
        fail = true;
      }
    }
    else if (sbRef.isSetPortRef() == true)
    {
      bool found = false;
      Port* port = plug->getPort(sbRef.getPortRef());
      if (port->isSetIdRef() == true)
      {
        if (plug->getSubmodel(port->getIdRef()) != NULL)
        {
          found = true;
        }
      }
      else if (port->isSetMetaIdRef() == true)
      {
        unsigned int i = 0;
        while (found == false &&  i < plug->getNumSubmodels())
        {
          if (port->getMetaIdRef() == plug->getSubmodel(i)->getMetaId())
          {
            found = true;
          }

          i++;
        }
      }

      if (found == false)
      {
        fail = true;
      }
    }
    else
    {
      // must be a metaidref
      std::string ref = sbRef.getMetaIdRef();
      bool found = false;
      unsigned int i = 0;
      while (found == false &&  i < plug->getNumSubmodels())
      {
        if (ref == plug->getSubmodel(i)->getMetaId())
        {
          found = true;
        }

        i++;
      }
      if (found == false)
      {
        fail = true;
      }
    }
  }
  else
  {
    fail = true;

    if (sbRef.isSetUnitRef() == true)
    {
      msg = "The 'unitRef' of a <sBaseRef>";
      msg += " is set to '";
      msg += sbRef.getUnitRef();
    }
    msg += "' which is not a submodel within the <model>.";
  }

  inv(fail == false);
}
END_CONSTRAINT


//20706 - caught at read
//20707 - caught at read
//20708 - caught at read
//20709 - caught at read
//20710 - caught at read
//20711 - caught at read

//20712
START_CONSTRAINT (CompSBaseRefMustReferenceObject, SBaseRef, sbRef)
{
  bool idRef = sbRef.isSetIdRef();
  bool unitRef = sbRef.isSetUnitRef();
  bool metaidRef = sbRef.isSetMetaIdRef();
  bool portRef  = sbRef.isSetPortRef();

  msg = "<sBaseRef> in ";
  const Model* mod = static_cast<const Model*>
                                    (sbRef.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (sbRef.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " does not refer to another object.";

  bool fail = true;

  if (idRef == true)
  {
    fail = false;
  }
  else if (unitRef == true)
  {
    fail = false;
  }
  else if (metaidRef == true)
  {
    fail = false;
  }
  else if (portRef == true)
  {
    fail = false;
  }

  inv(fail == false);


}
END_CONSTRAINT


//20713
START_CONSTRAINT (CompSBaseRefMustReferenceOnlyOneObject, SBaseRef, sbRef)
{

  bool idRef = sbRef.isSetIdRef();
  bool unitRef = sbRef.isSetUnitRef();
  bool metaidRef = sbRef.isSetMetaIdRef();
  bool portRef = sbRef.isSetPortRef();

  bool fail = false;

  msg = "<sBaseRef> in ";
  const Model* mod = static_cast<const Model*>
                                    (sbRef.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (sbRef.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to ";

  if (idRef == true)
  {
    msg += "object with id '";
    msg += sbRef.getIdRef();
    msg += "'";
    if (unitRef == true)
    {
      fail = true;
      msg += "and also unit with id '";
      msg += sbRef.getUnitRef();
      msg += "'";

      if ( metaidRef == true)
      {
        msg += "and also object with metaid '";
        msg += sbRef.getMetaIdRef();
        msg += "'";
      }

      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += sbRef.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += sbRef.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += sbRef.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += sbRef.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (unitRef == true)
  {
    msg += "unit with id '";
    msg += sbRef.getUnitRef();
    msg += "' and also ";
    
    if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += sbRef.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += sbRef.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += sbRef.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (metaidRef == true)
  {
    msg += "object with metaid '";
    msg += sbRef.getMetaIdRef();
    msg += "'";

    if (portRef == true)
    {
      fail = true;
      msg += "and also port with id '";
      msg += sbRef.getPortRef();
      msg += "'";
    }
    msg += ".";
  }

  inv(fail == false);

}
END_CONSTRAINT


//*************************************

//Port constraints

//20801
START_CONSTRAINT (CompPortMustReferenceObject, Port, p)
{
  pre (p.isSetId());

  bool idRef = p.isSetIdRef();
  bool unitRef = p.isSetUnitRef();
  bool metaidRef = p.isSetMetaIdRef();

  msg = "<port> '";
  msg += p.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                    (p.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (p.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " does not refer to another object.";

  bool fail = true;

  if (idRef == true)
  {
    fail = false;
  }
  else if (unitRef == true)
  {
    fail = false;
  }
  else if (metaidRef == true)
  {
    fail = false;
  }

  inv(fail == false);


}
END_CONSTRAINT


//20802
START_CONSTRAINT (CompPortMustReferenceOnlyOneObject, Port, p)
{
  pre (p.isSetId());

  bool idRef = p.isSetIdRef();
  bool unitRef = p.isSetUnitRef();
  bool metaidRef = p.isSetMetaIdRef();

  bool fail = false;

  msg = "<port> '";
  msg += p.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                    (p.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (p.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to ";

  if (idRef == true)
  {
    msg += "object with id '";
    msg += p.getIdRef();
    msg += "' and also ";
    if (unitRef == true)
    {
      fail = true;
      msg += "unit with id '";
      msg += p.getUnitRef();
      msg += "'";

      if ( metaidRef == true)
      {
        msg += "and also object with metaid '";
        msg += p.getMetaIdRef();
        msg += "'.";
      }
    }
    else if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += p.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (unitRef == true)
  {
    msg += "unit with id '";
    msg += p.getUnitRef();
    msg += "' and also ";
    
    if (metaidRef == true)
    {
      fail = true;
      msg += "object with metaid '";
      msg += p.getMetaIdRef();
      msg += "'.";
    }
  }

  inv(fail == false);

}
END_CONSTRAINT


//20803 - caught at read

//20804
EXTERN_CONSTRAINT( CompPortReferencesUnique, UniquePortReferences)

//*************************************

//Deletion constraints

//20901
START_CONSTRAINT (CompDeletionMustReferenceObject, Deletion, d)
{
  //pre (d.isSetId());

  bool idRef = d.isSetIdRef();
  bool unitRef = d.isSetUnitRef();
  bool metaidRef = d.isSetMetaIdRef();
  bool portRef  = d.isSetPortRef();

  msg = "<Deletion> '";
  msg += d.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                    (d.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (d.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " does not refer to another object.";

  bool fail = true;

  if (idRef == true)
  {
    fail = false;
  }
  else if (unitRef == true)
  {
    fail = false;
  }
  else if (metaidRef == true)
  {
    fail = false;
  }
  else if (portRef == true)
  {
    fail = false;
  }

  inv(fail == false);


}
END_CONSTRAINT


//20902
START_CONSTRAINT (CompDeletionMustReferOnlyOneObject, Deletion, d)
{
  //pre (d.isSetId());

  bool idRef = d.isSetIdRef();
  bool unitRef = d.isSetUnitRef();
  bool metaidRef = d.isSetMetaIdRef();
  bool portRef = d.isSetPortRef();

  bool fail = false;

  msg = "<Deletion> '";
  msg += d.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                    (d.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                     (d.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to ";

  if (idRef == true)
  {
    msg += "object with id '";
    msg += d.getIdRef();
    msg += "'";
    if (unitRef == true)
    {
      fail = true;
      msg += "and also unit with id '";
      msg += d.getUnitRef();
      msg += "'";

      if ( metaidRef == true)
      {
        msg += "and also object with metaid '";
        msg += d.getMetaIdRef();
        msg += "'";
      }

      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += d.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += d.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += d.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += d.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (unitRef == true)
  {
    msg += "unit with id '";
    msg += d.getUnitRef();
    msg += "' and also ";
    
    if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += d.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += d.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += d.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (metaidRef == true)
  {
    msg += "object with metaid '";
    msg += d.getMetaIdRef();
    msg += "'";

    if (portRef == true)
    {
      fail = true;
      msg += "and also port with id '";
      msg += d.getPortRef();
      msg += "'";
    }
    msg += ".";
  }

  inv(fail == false);

}
END_CONSTRAINT


//20903 - caught at read

//*************************************

//ReplacedElement constraints

//21001
START_CONSTRAINT (CompReplacedElementMustRefObject, ReplacedElement, repE)
{
  pre (repE.isSetSubmodelRef());

  bool idRef = repE.isSetIdRef();
  bool unitRef = repE.isSetUnitRef();
  bool metaidRef = repE.isSetMetaIdRef();
  bool portRef  = repE.isSetPortRef();
  bool deletion = repE.isSetDeletion();

  msg = "A <replacedElement> in ";
  const Model* mod = static_cast<const Model*>
                                  (repE.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                   (repE.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg = " does not refer to another object.";

  bool fail = true;

  if (idRef == true)
  {
    fail = false;
  }
  else if (unitRef == true)
  {
    fail = false;
  }
  else if (metaidRef == true)
  {
    fail = false;
  }
  else if (portRef == true)
  {
    fail = false;
  }
  else if (deletion == true)
  {
    fail = false;
  }

  inv(fail == false);


}
END_CONSTRAINT


//21002
START_CONSTRAINT (CompReplacedElementMustRefOnlyOne, ReplacedElement, repE)
{
  pre (repE.isSetSubmodelRef());

  bool idRef = repE.isSetIdRef();
  bool unitRef = repE.isSetUnitRef();
  bool metaidRef = repE.isSetMetaIdRef();
  bool portRef  = repE.isSetPortRef();
  bool deletion = repE.isSetDeletion();

  bool fail = false;

  msg = "<replacedElement> '";
  msg += repE.getId() ;
  msg += "' in ";
  const Model* mod = static_cast<const Model*>
                                (repE.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                  (repE.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to ";

  if (idRef == true)
  {
    msg += "object with id '";
    msg += repE.getIdRef();
    msg += "'";
    if (unitRef == true)
    {
      fail = true;
      msg += "and also unit with id '";
      msg += repE.getUnitRef();
      msg += "'";

      if ( metaidRef == true)
      {
        msg += "and also object with metaid '";
        msg += repE.getMetaIdRef();
        msg += "'";
      }

      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repE.getPortRef();
        msg += "'";
      }

      if (deletion == true)
      {
        msg += "and also deletion object '";
        msg += repE.getDeletion();
        msg += "'";
      }
      msg += ".";
    }
    else if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repE.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repE.getPortRef();
        msg += "'";
      }

      if (deletion == true)
      {
        msg += "and also deletion object '";
        msg += repE.getDeletion();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repE.getMetaIdRef();

      if (deletion == true)
      {
        msg += "and also deletion object '";
        msg += repE.getDeletion();
        msg += "'";
      }
      msg += "'.";
    }
    else if (deletion == true)
    {
      fail = true;
      msg += "and also deletion object '";
      msg += repE.getDeletion();
      msg += "'.";
    }
  }
  else if (unitRef == true)
  {
    msg += "unit with id '";
    msg += repE.getUnitRef();
    msg += "' and also ";
    
    if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repE.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repE.getPortRef();
        msg += "'";
      }
 
      if (deletion == true)
      {
        msg += "and also deletion object '";
        msg += repE.getDeletion();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repE.getMetaIdRef();
 
      if (deletion == true)
      {
        msg += "and also deletion object '";
        msg += repE.getDeletion();
        msg += "'";
      }
      msg += "'.";
    }
    else if (deletion == true)
    {
      fail = true;
      msg += "and also deletion object '";
      msg += repE.getDeletion();
      msg += "'.";
   }
  }
  else if (metaidRef == true)
  {
    msg += "object with metaid '";
    msg += repE.getMetaIdRef();
    msg += "'";

    if (portRef == true)
    {
      fail = true;
      msg += "and also port with id '";
      msg += repE.getPortRef();
      msg += "'";
    }
 
    if (deletion == true)
    {
      msg += "and also deletion object '";
      msg += repE.getDeletion();
      msg += "'";
    }
    msg += ".";
  }
  else if (portRef == true)
  {
    msg += "port with id '";
    msg += repE.getPortRef();
    msg += "'";

    if (deletion == true)
    {
      fail = true;
      msg += "and also deletion object '";
      msg += repE.getDeletion();
      msg += "'";
    }
     msg += ".";
  }

  inv(fail == false);

}
END_CONSTRAINT

//21003 - caught at read

//21004
START_CONSTRAINT (CompReplacedElementSubModelRef, ReplacedElement, repE)
{
  pre (repE.isSetSubmodelRef());

  msg = "The <replacedElement> refers to the submodel '";
  msg += repE.getSubmodelRef();
  msg += "' that is not part of the parent model.";

  bool fail = false;

  const CompModelPlugin * plug = 
                  static_cast<const CompModelPlugin*>(m.getPlugin("comp"));
  if (plug != NULL
    && plug->getSubmodel(repE.getSubmodelRef()) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT


//21005
START_CONSTRAINT (CompReplacedElementDeletionRef, ReplacedElement, repE)
{
  pre (repE.isSetSubmodelRef());
  pre (repE.isSetDeletion());

  msg = "A <replacedElement> in ";
  const Model* mod = static_cast<const Model*>
                                (repE.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                 (repE.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg = " refers to the deletion '";
  msg += repE.getDeletion();
  msg += "' that is not part of the parent model.";

  bool fail = false;

  const CompModelPlugin * plug = 
    static_cast<const CompModelPlugin*>(m.getPlugin("comp"));
  if (plug != NULL)
  {
    const Submodel * sub  = plug->getSubmodel(repE.getSubmodelRef());

    if (sub != NULL && sub->getDeletion(repE.getDeletion()) == NULL)
    {
      fail = true;
    }
  }

  inv(fail == false);
}
END_CONSTRAINT

//21006
START_CONSTRAINT (CompReplacedElementConvFactorRef, ReplacedElement, repE)
{
  pre (repE.isSetSubmodelRef());
  pre (repE.isSetConversionFactor());

  msg = "The 'conversionFactor' of a <replacedElement> in ";
  const Model* mod = static_cast<const Model*>
                                 (repE.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                  (repE.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg = " is set to '";
  msg += repE.getConversionFactor();
  msg += "' which is not a <parameter> within the model.";

  bool fail = false;

  if (m.getParameter(repE.getConversionFactor()) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

//21007 - repeat of 10308
//21008 - repeat of 10309
//21009 - repeat of 10310

//21010
EXTERN_CONSTRAINT( CompReplacedElementSameReference, UniqueReplacedReferences)

//21011
START_CONSTRAINT (CompReplacedElementNoDelAndConvFact, ReplacedElement, repE)
{
  pre (repE.isSetDeletion());

  bool fail = false;

  if (repE.isSetConversionFactor() == true)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

//*************************************

//ReplacedBy constraints

//21101
START_CONSTRAINT (CompReplacedByMustRefObject, ReplacedBy, repBy)
{
  pre (repBy.isSetSubmodelRef());

  bool idRef = repBy.isSetIdRef();
  bool unitRef = repBy.isSetUnitRef();
  bool metaidRef = repBy.isSetMetaIdRef();
  bool portRef = repBy.isSetPortRef();

  msg = "A <replacedBy> in ";
  const Model* mod = static_cast<const Model*>
                                (repBy.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                 (repBy.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " does not refer to another object.";

  bool fail = true;

  if (idRef == true)
  {
    fail = false;
  }
  else if (unitRef == true)
  {
    fail = false;
  }
  else if (metaidRef == true)
  {
    fail = false;
  }
  else if (portRef == true)
  {
    fail = false;
  }

  inv(fail == false);
}
END_CONSTRAINT


//21102
START_CONSTRAINT (CompReplacedByMustRefOnlyOne, ReplacedBy, repBy)
{
  pre (repBy.isSetSubmodelRef());

  bool idRef = repBy.isSetIdRef();
  bool unitRef = repBy.isSetUnitRef();
  bool metaidRef = repBy.isSetMetaIdRef();
  bool portRef = repBy.isSetPortRef();

  bool fail = false;

  msg = "A <replacedBy> object in ";
  const Model* mod = static_cast<const Model*>
                                (repBy.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                 (repBy.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to ";

  if (idRef == true)
  {
    msg += "object with id '";
    msg += repBy.getIdRef();
    msg += "'";
    if (unitRef == true)
    {
      fail = true;
      msg += "and also unit with id '";
      msg += repBy.getUnitRef();
      msg += "'";

      if ( metaidRef == true)
      {
        msg += "and also object with metaid '";
        msg += repBy.getMetaIdRef();
        msg += "'";
      }

      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repBy.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repBy.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repBy.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repBy.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (unitRef == true)
  {
    msg += "unit with id '";
    msg += repBy.getUnitRef();
    msg += "' and also ";
    
    if (metaidRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repBy.getMetaIdRef();
      msg += "'";
 
      if (portRef == true)
      {
        msg += "and also port with id '";
        msg += repBy.getPortRef();
        msg += "'";
      }
      msg += ".";
    }
    else if (portRef == true)
    {
      fail = true;
      msg += "and also object with metaid '";
      msg += repBy.getMetaIdRef();
      msg += "'.";
    }
  }
  else if (metaidRef == true)
  {
    msg += "object with metaid '";
    msg += repBy.getMetaIdRef();
    msg += "'";

    if (portRef == true)
    {
      fail = true;
      msg += "and also port with id '";
      msg += repBy.getPortRef();
      msg += "'";
    }
    msg += ".";
  }

  inv(fail == false);

}
END_CONSTRAINT

//21103 - caught at read

//21104
START_CONSTRAINT (CompReplacedBySubModelRef, ReplacedBy, repBy)
{
  pre (repBy.isSetSubmodelRef());

  msg = "A <replacedBy> in ";
  const Model* mod = static_cast<const Model*>
                                (repBy.getAncestorOfType(SBML_MODEL, "core"));
  if (mod == NULL) {
    mod = static_cast<const Model*>
                   (repBy.getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  }
  if (mod == NULL || !mod->isSetId()) {
    msg += "the main model in the document";
  }
  else {
    msg += "the model '";
    msg += mod->getId();
    msg += "'";
  }
  msg += " refers to the submodel '";
  msg += repBy.getSubmodelRef();
  msg += "' that is not part of the parent model.";

  bool fail = false;

  const CompModelPlugin * plug = 
                      static_cast<const CompModelPlugin*>(m.getPlugin("comp"));
  if (plug != NULL
    && plug->getSubmodel(repBy.getSubmodelRef()) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

//21201
EXTERN_CONSTRAINT( CompMustReplaceSameClass, ClassReplacements)

//21202 - caught during flattening
//21203 - caught during flattening

//21204
EXTERN_CONSTRAINT(CompMustReplacePackageIDs, PackageIdReplacementCheck)

// 90115 (note not on a replacedBy)
// 90115 - port
START_CONSTRAINT (CompIdRefMayReferenceUnknownPackage, Port, p)
{
  pre(p.isSetIdRef());

  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);
  
  msg = "The 'idRef' of a <port>";
  msg += " is set to '";
  msg += p.getIdRef();
  msg += "' which is not an element within the <model>.";
  msg += " However it may be an identifier of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  ReferencedModel *ref = new ReferencedModel(m, p);
  const Model* mod = ref->getReferencedModel();
  
  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  inv(mIds.contains(p.getIdRef()))
}
END_CONSTRAINT


// 90115 - deletion
START_CONSTRAINT (CompIdRefMayReferenceUnknownPackage, Deletion, d)
{
  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);
  pre(d.isSetIdRef());
  
  const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'idRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'. However it may be an identifier of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* mod = ref->getReferencedModel();
  
  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  inv(mIds.contains(d.getIdRef()))
}
END_CONSTRAINT


// 90115 - replacedElement
START_CONSTRAINT (CompIdRefMayReferenceUnknownPackage, ReplacedElement, repE)
{
  pre(repE.isSetIdRef());
  pre(repE.isSetSubmodelRef());

  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);
  
  msg = "The 'idRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'. However it may be an identifier of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  ReferencedModel *ref = new ReferencedModel(m, repE);
  const Model* mod = ref->getReferencedModel();
  
  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  inv(mIds.contains(repE.getIdRef()))
}
END_CONSTRAINT


// 90115 - sBaseRef
START_CONSTRAINT (CompIdRefMayReferenceUnknownPackage, SBaseRef, sbRef)
{
  pre(sbRef.isSetIdRef());

  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);
  
  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'idRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getIdRef();
  msg += "' which is not an element within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }
  msg += "However it may be an identifier of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  IdFilter filter;

  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* mod = ref->getReferencedModel();
  
  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getId());
  }

  delete allElements;

  inv(mIds.contains(sbRef.getIdRef()))
}
END_CONSTRAINT


// 90116
//90116 - port
START_CONSTRAINT (CompMetaIdRefMayReferenceUnknownPkg, Port, p)
{
  pre(p.isSetMetaIdRef());
  
  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);

  msg = "The 'metaIdRef' of a <port>";
  msg += " is set to '";
  msg += p.getMetaIdRef();
  msg += "' which is not an element within the <model>. ";
  msg += "However it may be the 'metaid' of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  ReferencedModel *ref = new ReferencedModel(m, p);
  const Model* mod = ref->getReferencedModel();

  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  inv(mIds.contains(p.getMetaIdRef()))
}
END_CONSTRAINT

// 90116 - deletion
START_CONSTRAINT (CompMetaIdRefMayReferenceUnknownPkg, Deletion, d)
{
  pre(d.isSetMetaIdRef());
  
  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);

  const Submodel * sub = static_cast<const Submodel*>
                        (d.getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
  pre (sub != NULL);

  msg = "The 'metaIdRef' of a <deletion>";
  msg += " is set to '";
  msg += d.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += sub->getId();
  msg += "'. ";
  msg += "However it may be the 'metaid' of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  ReferencedModel *ref = new ReferencedModel(m, d);
  const Model* mod = ref->getReferencedModel();

  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  inv(mIds.contains(d.getMetaIdRef()))
}
END_CONSTRAINT

// 90116 - replacedElement
START_CONSTRAINT (CompMetaIdRefMayReferenceUnknownPkg, ReplacedElement, repE)
{
  pre(repE.isSetMetaIdRef());
  pre(repE.isSetSubmodelRef());

  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);

  msg = "The 'metaidRef' of a <replacedElement>";
  msg += " is set to '";
  msg += repE.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";
  msg += "submodel '";
  msg += repE.getSubmodelRef();
  msg += "'. ";
  msg += "However it may be the 'metaid' of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  ReferencedModel *ref = new ReferencedModel(m, repE);
  const Model* mod = ref->getReferencedModel();

  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  inv(mIds.contains(repE.getMetaIdRef()))
}
END_CONSTRAINT

// 90116 - sBaseRef
START_CONSTRAINT (CompMetaIdRefMayReferenceUnknownPkg, SBaseRef, sbRef)
{
  pre(sbRef.isSetMetaIdRef());

  /* only log this if there are unknown packages present */
  const SBMLDocument *doc = m.getSBMLDocument();
  SBMLErrorLog *errlog = const_cast<SBMLErrorLog*>(doc->getErrorLog());
  bool unknownPackagePresent = false;
  if (errlog->contains(UnrequiredPackagePresent)
    || errlog->contains(RequiredPackagePresent))
  {
    unknownPackagePresent = true;
  }
  pre ( unknownPackagePresent == true);

  pre (sbRef.getParentSBMLObject() != NULL);

  int tc = sbRef.getParentSBMLObject()->getTypeCode();

  msg = "The 'metaIdRef' of a <sBaseRef>";
  msg += " is set to '";
  msg += sbRef.getMetaIdRef();
  msg += "' which is not an element within the <model> referenced by ";

  if (tc == SBML_COMP_REPLACEDELEMENT)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedElement*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_REPLACEDBY)
  {
    msg += "the submodel '";
    msg += static_cast<const ReplacedBy*>(sbRef.getParentSBMLObject())
                                               ->getSubmodelRef();
    msg += "'.";
  }
  else if (tc == SBML_COMP_PORT)
  {
    msg += "port '";
    msg += sbRef.getParentSBMLObject()->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_DELETION)
  {
    const Submodel * sub = static_cast<const Submodel*>
                           (sbRef.getParentSBMLObject()
                           ->getAncestorOfType(SBML_COMP_SUBMODEL, "comp"));
    pre (sub != NULL);
    
    msg += "the submodel '";
    msg += sub->getId();
    msg += "'.";
  }
  else if (tc == SBML_COMP_SBASEREF)
  {
    msg += "the parent sBaseRef.";
  }
  msg += " However it may be the 'metaid' of an object within an ";
  msg += "unrecognised package. ";

  IdList mIds;

  // create the filter we want to use
  MetaIdFilter filter;

  //  get a list of all elements with an id
  ReferencedModel *ref = new ReferencedModel(m, sbRef);
  const Model* mod = ref->getReferencedModel();

  pre (mod != NULL);
  
  List* allElements = const_cast<Model*>(mod)->getAllElements(&filter);

  for (unsigned int i = 0; i < allElements->getSize(); i++)
  {
    mIds.append(static_cast<SBase*>(allElements->get(i))->getMetaId());
  }

  delete allElements;

  inv(mIds.contains(sbRef.getMetaIdRef()))
}
END_CONSTRAINT



/** @endcond */


/**
 * @file    SBMLHttpResolverExample.cpp
 * @brief   Example of the definition of an SBMLHttpResolver, and how one would 
 *          use it.
 * @author  Frank Bergmann
 *
 * <!--------------------------------------------------------------------------
 * This sample program is distributed under a different license than the rest
 * of libSBML.  This program uses the open-source MIT license, as follows:
 *
 * Copyright (c) 2013-2014 by the California Institute of Technology
 * (California, USA), the European Bioinformatics Institute (EMBL-EBI, UK)
 * and the University of Heidelberg (Germany), with support from the National
 * Institutes of Health (USA) under grant R01GM070923.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Neither the name of the California Institute of Technology (Caltech), nor
 * of the European Bioinformatics Institute (EMBL-EBI), nor of the University
 * of Heidelberg, nor the names of any contributors, may be used to endorse
 * or promote products derived from this software without specific prior
 * written permission.
 * ------------------------------------------------------------------------ -->
 *
 * @class SBMLHttpResolver
 * @brief resolves HTTP documents, based on 
 * http://stackoverflow.com/questions/9786150/save-curl-content-result-into-a-string-in-c
 */

#include <iostream>
#include <string>

#include <curl/curl.h>

#include <sbml/SBMLTypes.h>
#include <sbml/packages/comp/common/CompExtensionTypes.h>
#include <sbml/packages/comp/util/SBMLResolver.h>
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>
#include <sbml/packages/comp/util/SBMLUri.h>

using namespace std;


static size_t WriteCallback(void *contents, size_t size,
                            size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}


class LIBSBML_EXTERN SBMLHttpResolver : public SBMLResolver
{
public:

    /**
     * Creates a new SBMLHttpResolver object.
     */
    SBMLHttpResolver ()
    {
    }


    /**
     * Copy constructor; creates a copy of an SBMLHttpResolver object.
     *
     * @param c the SBMLHttpResolver object to copy.
     *
     * @throws @if python ValueError @else SBMLConstructorException @endif@~
     * Thrown if the argument @p orig is @c NULL.
     */
    SBMLHttpResolver(const SBMLHttpResolver& c)
    {

    }


    /**
     * Destroy this SBMLHttpResolver object.
     */
    virtual ~SBMLHttpResolver ()
    {
    }


    /**
     * Assignment operator for SBMLHttpResolver.
     *
     * @param rhs The object whose values are used as the basis of the
     * assignment.
     *
     * @throws @if python ValueError @else SBMLConstructorException @endif@~
     * Thrown if the argument @p rhs is @c NULL.
     */
    SBMLHttpResolver& operator=(const SBMLHttpResolver& rhs)
    {
        return *this;
    }


    /**
     * Creates and returns a deep copy of this SBMLHttpResolver object.
     *
     * @return a (deep) copy of this SBMLHttpResolver object.
     */
    virtual SBMLHttpResolver* clone() const
    {
        return new SBMLHttpResolver(*this);
    }

    /**
     * resolve the document for the given uri
     *
     * @param uri the uri to the target document
     * @param baseUri base uri, in case the uri is a relative one
     *
     * @return  the document, if this resolver can resolve the document or NULL.
     */
    virtual SBMLDocument* resolve(const std::string &uri,
                                  const std::string& baseUri="") const
    {
        SBMLUri * resolved = resolveUri(uri, baseUri);
        if (resolved == NULL)
            return NULL;

        CURL *curl;
        CURLcode res;
        std::string readBuffer;

        curl = curl_easy_init();
        if(curl) 
        {
            curl_easy_setopt(curl, CURLOPT_URL, resolved->getUri().c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
            res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);

            if (res == CURLE_OK)
                return readSBMLFromString(readBuffer.c_str());

            // here is where errors could be handled, like
            // as would occur whenever a network issue arrises.      
        }

        // something went wrong ... 
        return NULL;
    }


    /**
     * resolve the full uri for the given uri without actually reading the document
     *
     * @param uri the uri to the target document
     * @param baseUri base uri, in case the uri is a relative one
     *
     * @return  the full uri to the document, if this resolver can resolve the document or NULL.
     */
    virtual SBMLUri* resolveUri(const std::string &sUri,
                                const std::string& sBaseUri="") const
    {
        SBMLUri uri(sUri);

        if (uri.getScheme() != "http")
        {
            // this example just displays how to resolve http uris, so if we don't have one of those, 
            // bail
            return NULL;
        }

        // here we could get fancy and construct relative uris ... but for 
        // now this is good enough :) 

        return new SBMLUri(uri);
    }


protected:

private:

};


int resolveModelFromUri(const std::string& uri)
{
    // create an external document to test
    SBMLDocument doc(new CompPkgNamespaces());
    CompSBMLDocumentPlugin* plugin = (CompSBMLDocumentPlugin*)doc.getPlugin("comp");
  
    // create an external model definition 
    ExternalModelDefinition *external = plugin->createExternalModelDefinition();
    external->setSource(uri);

    Model* model = external->getReferencedModel();
    if (model == NULL)
    {
        // bummer something went wrong ... 
        return -1;
    }

    // do something with the model
    if (model->isSetId())
        cout << "resolved: " << model->getId() << endl;
    else
        cout << "resolved: " << uri << endl;

    if (model->isSetNotes())
        cout << model->getNotesString() << endl << endl;

    // all good
    return 0;
}

int main(void)
{
    // create http resolver 
    SBMLHttpResolver resolver;

    // the resolver can of course already be used: 

    SBMLDocument* originalBM1 =
        resolver.resolve("http://www.ebi.ac.uk/biomodels/models-main/publ/BIOMD0000000001/BIOMD0000000001.xml.origin");
    if (originalBM1 != NULL)
    {
        if (originalBM1->getModel()->isSetId())
            cout << "resolved: " << originalBM1->getModel()->getId() << endl;
        else 
            cout << "resolved: " << "BIOMD0000000001.xml.origin" << endl;
    }
    else
    {
        cout << "an error occured while trying to resolve the document. " << endl;
    }

    // but usually what one would do would be to 
    // add it to the registry
    SBMLResolverRegistry::getInstance().addResolver(&resolver);

    // and then go from there ...
    if (resolveModelFromUri("http://www.ebi.ac.uk/biomodels/models-main/publ/BIOMD0000000001/BIOMD0000000001.xml.origin") != 0 )
    {
        cout << "Comp V1 does not support L2V1 documents!" << endl;
    }

    if (resolveModelFromUri("http://www.ebi.ac.uk/biomodels-main/download?mid=BMID000000063853") != 0 )
    {
        cout << "something went wrong while getting the l3v1." << endl;
    }

    if (resolveModelFromUri("http://sbml.svn.sourceforge.net/viewvc/sbml/branches/libsbml-packages/comp/src/sbml/packages/comp/util/test/test-data/subdir/new_aggregate.xml?revision=16695") != 0 )
    {
        cout << "something went wrong while getting the l3v1 comp model." << endl;
    }

    cout << "done!";

    return 0;
}

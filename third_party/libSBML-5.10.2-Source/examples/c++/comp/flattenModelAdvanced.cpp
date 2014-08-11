/**
 * @file    flattenModelAdvanced.cpp
 * @brief   advanced SBML hierarchical composition flattening example
 *          if an SBase element has a name, this name will also hint 
 *          where the element came from. 
 *           
 * @author  Sarah Keating
 * @author  Frank T. Bergmann
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
 */

#include <iostream>
#include <sstream>

#include <sbml/SBMLTypes.h>
#include <sbml/extension/SBMLExtensionRegister.h>
#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/util/PrefixTransformer.h>

#ifdef  LIBSBML_HAS_PACKAGE_COMP
#include <sbml/packages/comp/common/CompExtensionTypes.h>
#else
#error This example requires libSBML to be compiled with support for COMP.
#endif


LIBSBML_CPP_NAMESPACE_USE
using namespace std;

const string usage = "Usage: flattenModelAdvanced [-p] inputFile outputFile\n"
" -p : list unused ports";


class CPrefixNameTransformer : public PrefixTransformer
{
public:
	CPrefixNameTransformer() {}

	static void replaceStringInPlace(std::string& subject, const std::string& search,
	const std::string& replace)
	{
		size_t pos = 0;

		while ((pos = subject.find(search, pos)) != std::string::npos)
		{
			subject.replace(pos, search.length(), replace);
			pos += replace.length();
		}
	}

	static inline std::string &rtrim(std::string &str)
	{
		size_t endpos = str.find_last_not_of(" \t");

		if (string::npos != endpos)
		{
			str = str.substr(0, endpos + 1);
		}

		return str;
	}

	const std::string& cleanName(std::string& prefix)
	{
		std::replace(prefix.begin(), prefix.end(), '_', ' ');
		replaceStringInPlace(prefix, "  ", " ");
		rtrim(prefix);
		return prefix;
	}

	virtual int transform(SBase* element)
	{
		if (element == NULL || getPrefix().empty())
		return LIBSBML_OPERATION_SUCCESS;

		// set up ids
    // this will rename the SIds and SIdRefs as
    // the comp flattening routine does by default
		PrefixTransformer::transform(element);

		// skip local parameters, as they are not renamed
		if (element->getTypeCode() == SBML_LOCAL_PARAMETER)
		return LIBSBML_OPERATION_SUCCESS;

		// setup names
    // here we want to rename the name attribute to also indicate if
    // the element came from a subModel
		if (element->isSetName())
		{
			std::stringstream newName;
			std::string prefix = getPrefix();
			newName << element->getName() << " (" << cleanName(prefix) << ")";
			element->setName(newName.str());
		}

		return LIBSBML_OPERATION_SUCCESS;
	}
};


int main(int argc,char** argv)
{
	bool leavePorts = false;
	if (argc < 3)
	{
		cout << usage << endl;
		return 1;
	}
	else if (argc == 3)
	{
		if ( string("-p") == string(argv[1]) )
		{
			cout << usage << endl;
			return 1;
		}       
	}

	int  argIndex = 1;

	if ( string("-p") == string(argv[1]) )
	{
		leavePorts = true;
		++argIndex;
	}     

	if (SBMLExtensionRegistry::isPackageEnabled("comp") == false)
	{
		cerr << "The version of libsbml being used does not have the comp"
		     << " package code enabled" << endl;
		return 1;
	}

	const char* inputFile   = argv[argIndex];
	const char* outputFile  = argv[argIndex+1];

	SBMLDocument* document = readSBML(inputFile);

	if (document->getNumErrors() > 0)
	{
		cerr << "Encountered the following SBML errors:" << endl;
		document->printErrors(cerr);
		return 1;
	}

	// add the advanced prefix transformer to the comp plugin
  // this transformer will be used during the flattening process
  // to rename SIds and SIdRefs (as would happen by default)
  // but allows the user to specify other changes that they
  // may wsh to be made (e.g. rename the name attribute)
	CompModelPlugin* mPlug = dynamic_cast<CompModelPlugin*>(document->getModel()->getPlugin("comp"));
	CPrefixNameTransformer trans;

	if (mPlug != NULL)
	{
		mPlug->setTransformer(&trans);
	}

	ConversionProperties* props = new ConversionProperties();

	props->addOption("flatten comp");
	props->addOption("leavePorts", leavePorts);

	SBMLConverter* converter = 
	SBMLConverterRegistry::getInstance().getConverterFor(*props);


	converter->setDocument(document);

	int result = converter->convert();

	if (result != LIBSBML_OPERATION_SUCCESS)
	{
		cerr << "Conversion failed\n";
		document->printErrors();
	}

	// unset the transformer before it goes out of scope
	if (mPlug != NULL)
	{
		mPlug->setTransformer(NULL);
	}

	writeSBML(document, outputFile);

	delete converter;
	delete document;

}

/**
 * @file    FlattenModel.cs
 * @brief   Flattens the comp code from the given SBML file.
 * @author  Lucian Smith
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

using System;
using libsbmlcs;

namespace CompExamples
{
    public class SpecExample2
    {
        private static int Main(string[] args)
        {
            var sbmlns = new SBMLNamespaces(3, 1, "comp", 1);

            // create the document
            var document = new SBMLDocument(sbmlns);

            //Define the external model definitions
            var compdoc = (CompSBMLDocumentPlugin)(document.getPlugin("comp"));
            compdoc.setRequired(true);
            var extmod = compdoc.createExternalModelDefinition();
            extmod.setId("ExtMod1");
            extmod.setSource("enzyme_model.xml");
            extmod.setModelRef("enzyme");


            // create the main Model
            var model = document.createModel();

            // Set the submodels
            var mplugin = (CompModelPlugin)(model.getPlugin("comp"));
            var submod1 = mplugin.createSubmodel();
            submod1.setId("A");
            submod1.setModelRef("ExtMod1");
            var submod2 = mplugin.createSubmodel();
            submod2.setId("B");
            submod2.setModelRef("ExtMod1");

            // create a replacement compartment
            var comp = model.createCompartment();
            comp.setSpatialDimensions(3);
            comp.setConstant(true);
            comp.setId("comp");
            comp.setSize(1L);

            //Tell the model that this compartment replaces both of the inside ones.
            var compartplug = (CompSBasePlugin)(comp.getPlugin("comp"));
            var re = new ReplacedElement();
            re.setIdRef("comp");
            re.setSubmodelRef("A");
            compartplug.addReplacedElement(re);
            re.setSubmodelRef("B");
            compartplug.addReplacedElement(re);

            // create a replacement species
            var spec = model.createSpecies();
            spec.setCompartment("comp");
            spec.setHasOnlySubstanceUnits(false);
            spec.setConstant(false);
            spec.setBoundaryCondition(false);
            spec.setId("S");

            //Tell the model that this species replaces both of the inside ones.
            var spp = (CompSBasePlugin)(spec.getPlugin("comp"));
            re.setIdRef("S");
            re.setSubmodelRef("A");
            spp.addReplacedElement(re);
            re.setSubmodelRef("B");
            spp.addReplacedElement(re);


            libsbml.writeSBMLToFile(document, "spec_example2.xml");
            document = libsbml.readSBMLFromFile("spec_example2.xml");
            if (document == null)
            {
                Console.WriteLine("Error reading back in file.");
                return -1;
            }
            else
            {
                document.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, false);
                document.checkConsistency();
                if (document.getErrorLog().getNumFailsWithSeverity(2) > 0 ||
                    document.getErrorLog().getNumFailsWithSeverity(3) > 0)
                {
                    var stream = new OStringStream();
                    document.printErrors(stream);
                    Console.WriteLine("Errors encoutered when round-tripping  SBML file: \n" +
                                      stream.str());
                    return -1;
                }
                libsbml.writeSBMLToFile(document, "spec_example2_rt.xml");
            }

            return 0;
        }
    }
}

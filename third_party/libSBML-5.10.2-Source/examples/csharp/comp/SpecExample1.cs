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
    public class SpecExample1
    {
        private static int Main(string[] args)
        {
            var retval = 0;
            var sbmlns = new SBMLNamespaces(3, 1, "comp", 1);

            // create the document
            var document = new SBMLDocument(sbmlns);

            //Create our submodel
            var compdoc = (CompSBMLDocumentPlugin) (document.getPlugin("comp"));
            compdoc.setRequired(true);
            var mod1 = compdoc.createModelDefinition();
            mod1.setId("enzyme");
            mod1.setName("enzyme");
            var comp = mod1.createCompartment();
            comp.setSpatialDimensions(3);
            comp.setConstant(true);
            comp.setId("comp");
            comp.setSize(1L);
            var spec = new Species(3, 1);
            spec.setCompartment("comp");
            spec.setHasOnlySubstanceUnits(false);
            spec.setConstant(false);
            spec.setBoundaryCondition(false);
            spec.setId("S");
            mod1.addSpecies(spec);
            spec.setId("E");
            mod1.addSpecies(spec);
            spec.setId("D");
            mod1.addSpecies(spec);
            spec.setId("ES");
            mod1.addSpecies(spec);
            var rxn = new Reaction(3, 1);
            rxn.setReversible(true);
            rxn.setFast(false);
            var rxn2 = new Reaction(rxn);
            rxn.setId("J0");
            rxn2.setId("J1");
            var sr = new SpeciesReference(3, 1);
            sr.setConstant(true);
            sr.setStoichiometry(1);
            sr.setSpecies("S");
            rxn.addReactant(sr);
            sr.setSpecies("E");
            rxn.addReactant(sr);
            rxn2.addProduct(sr);
            sr.setSpecies("ES");
            rxn.addProduct(sr);
            rxn2.addReactant(sr);
            sr.setSpecies("D");
            rxn2.addProduct(sr);

            mod1.addReaction(rxn);
            mod1.addReaction(rxn2);

            // create the Model
            var model = document.createModel();
            model.setId("aggregate");

            // Create a submodel
            var mplugin = (CompModelPlugin) (model.getPlugin("comp"));
            var submod1 = mplugin.createSubmodel();
            submod1.setId("submod1");
            submod1.setModelRef("enzyme");

            var submod2 = new Submodel();
            submod2.setId("submod2");
            submod2.setModelRef("enzyme");
            mplugin.addSubmodel(submod2);

            libsbml.writeSBMLToFile(document, "enzyme_model.xml");
            document = libsbml.readSBMLFromFile("enzyme_model.xml");
            if (document == null)
            {
                Console.WriteLine("Error reading back in file.");
                retval = -1;
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
                    retval = -1;
                }
                libsbml.writeSBMLToFile(document, "enzyme_model_rt.xml");
            }
            return retval;
        }
    }
}

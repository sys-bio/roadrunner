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
    public class SpecExample3
    {
        private static int Main(string[] args)
        {
            var retval = 0;
            var sbmlns = new SBMLNamespaces(3, 1, "comp", 1);

            // create the document
            var document = new SBMLDocument(sbmlns);

            //Define the external model definition
            var compdoc = (CompSBMLDocumentPlugin)(document.getPlugin("comp"));
            compdoc.setRequired(true);
            var extmod = compdoc.createExternalModelDefinition();
            extmod.setId("ExtMod1");
            extmod.setSource("enzyme_model.xml");
            extmod.setModelRef("enzyme");

            //Define the 'simple' model
            var mod1 = compdoc.createModelDefinition();
            mod1.setId("simple");
            var comp = mod1.createCompartment();
            comp.setSpatialDimensions(3);
            comp.setConstant(true);
            comp.setId("comp");
            comp.setSize(1L);

            var spec = new Species(sbmlns);
            //We have to construct it this way because we get the comp plugin from it later.
            spec.setCompartment("comp");
            spec.setHasOnlySubstanceUnits(false);
            spec.setConstant(false);
            spec.setBoundaryCondition(false);
            spec.setId("S");
            spec.setInitialConcentration(5);
            mod1.addSpecies(spec);
            spec.setId("D");
            spec.setInitialConcentration(10);
            mod1.addSpecies(spec);

            var rxn = new Reaction(3, 1);
            rxn.setReversible(true);
            rxn.setFast(false);
            rxn.setId("J0");

            var sr = new SpeciesReference(3, 1);
            sr.setConstant(true);
            sr.setStoichiometry(1);
            sr.setSpecies("S");
            rxn.addReactant(sr);
            sr.setSpecies("D");
            rxn.addProduct(sr);

            mod1.addReaction(rxn);

            var mod1plug = (CompModelPlugin)(mod1.getPlugin("comp"));
            var port = new Port();
            port.setId("S_port");
            port.setIdRef("S");
            mod1plug.addPort(port);

            var port2 = mod1plug.createPort();
            port2.setId("D_port");
            port2.setIdRef("D");

            port.setId("comp_port");
            port.setIdRef("comp");
            mod1plug.addPort(port);

            port.setId("J0_port");
            port.setIdRef("J0");
            mod1plug.addPort(port);

            // create the Model
            var model = document.createModel();
            model.setId("complexified");

            // Set the submodels
            var mplugin = (CompModelPlugin)(model.getPlugin("comp"));
            var submod1 = mplugin.createSubmodel();
            submod1.setId("A");
            submod1.setModelRef("ExtMod1");
            var submod2 = mplugin.createSubmodel();
            submod2.setId("B");
            submod2.setModelRef("simple");
            var del = submod2.createDeletion();
            del.setPortRef("J0_port");

            // Synchronize the compartments
            var mcomp = model.createCompartment();
            mcomp.setSpatialDimensions(3);
            mcomp.setConstant(true);
            mcomp.setId("comp");
            mcomp.setSize(1L);
            var compartplug = (CompSBasePlugin)(mcomp.getPlugin("comp"));
            var re = new ReplacedElement();
            re.setIdRef("comp");
            re.setSubmodelRef("A");
            compartplug.addReplacedElement(re);
            re.setSubmodelRef("B");
            re.unsetIdRef();
            re.setPortRef("comp_port");
            compartplug.addReplacedElement(re);

            //Synchronize the species
            spec.setId("S");
            spec.setInitialConcentration(5);
            var specplug = (CompSBasePlugin)(spec.getPlugin("comp"));
            var sre = specplug.createReplacedElement();
            sre.setSubmodelRef("A");
            sre.setIdRef("S");
            var sre2 = specplug.createReplacedElement();
            sre2.setSubmodelRef("B");
            sre2.setPortRef("S_port");
            model.addSpecies(spec);

            spec.setId("D");
            spec.setInitialConcentration(10);
            sre.setIdRef("D");
            sre2.setPortRef("D_port");
            model.addSpecies(spec);

            libsbml.writeSBMLToFile(document, "spec_example3.xml");
            document = libsbml.readSBMLFromFile("spec_example3.xml");
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
                libsbml.writeSBMLToFile(document, "spec_example3_rt.xml");
            }
            return retval;
        }
    }
}

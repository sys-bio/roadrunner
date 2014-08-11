/**
 * @file    FlattenModel.cs
 * @brief   Flattens the comp code from the given SBML file.
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

using System;
using libsbmlcs;

namespace CompExamples
{
  public class CompExample1
  {
    public static int Main(string[] args)
    {
      int retval = 0;
      SBMLNamespaces sbmlns = new SBMLNamespaces(3, 1, "comp", 1);

      // create the document
      SBMLDocument document = new SBMLDocument(sbmlns);

      // create the Model
      Model model = document.createModel();

      // create a replacement parameter
      Parameter parameter = model.createParameter();
      parameter.setId("x");
      parameter.setConstant(true);

      // create a parameter to be a conversion factor
      Parameter param2 = model.createParameter();
      param2.setId("x_conv");
      param2.setMetaId("_110013");
      param2.setConstant(true);

      // create a parameter to be a conversion factor
      Parameter param3 = model.createParameter();
      param3.setId("lcf");
      param3.setConstant(true);

      // Convert parameter to the plugin version so we can add the new attributes and replacements to it.
      CompSBasePlugin splugin = (CompSBasePlugin) (parameter.getPlugin("comp"));

      // Add a replaced element.
      ReplacedElement rep1 = splugin.createReplacedElement();
      int rv = rep1.setSubmodelRef("submod1");
      rv = rep1.setConversionFactor("x_conv");
      rv = rep1.setIdRef("param1");

      // Add a second replaced element in a different way.
      ReplacedElement rep2 = new ReplacedElement();
      rv = rep2.setSubmodelRef("submod2");
      rv = rep2.setDeletion("del1");
      rv = splugin.addReplacedElement(rep2);

      //Now create a replaced element that points into a submodel.
      rep2.unsetDeletion();
      rep2.setIdRef("submod2");
      SBaseRef sbr5 = rep2.createSBaseRef();
      sbr5.setIdRef("submodelG");
      SBaseRef sbr6 = sbr5.createSBaseRef();
      sbr6.setIdRef("buriedElement");
      splugin.addReplacedElement(rep2);


      // Create a submodel
      CompModelPlugin mplugin = (CompModelPlugin) (model.getPlugin("comp"));
      CompSBMLDocumentPlugin compdoc = (CompSBMLDocumentPlugin) (document.getPlugin("comp"));
      compdoc.setRequired(true);

      ModelDefinition moddef1 = compdoc.createModelDefinition();
      moddef1.setId("Mod1");
      Parameter m1param1 = moddef1.createParameter();
      m1param1.setId("param1");
      m1param1.setConstant(true);
      Parameter m1param2 = moddef1.createParameter();
      m1param2.setId("param2");
      m1param2.setConstant(false);
      m1param2.setValue(3.2);

      ModelDefinition moddef2 = new ModelDefinition();
      moddef2.setId("Mod2");
      Parameter subparam2 = moddef2.createParameter();
      subparam2.setId("subparam2");
      subparam2.setConstant(false);
      compdoc.addModelDefinition(moddef2);


      ExternalModelDefinition extmod1 = compdoc.createExternalModelDefinition();
      extmod1.setId("ExtMod1");
      extmod1.setSource("urn:miriam:biomodels.db:BIOMD0000000127");

      ExternalModelDefinition extmod2 = new ExternalModelDefinition();
      extmod2.setId("ExtMod2");
      extmod2.setSource("otherfile.xml");
      extmod2.setModelRef("modelnamethere");
      extmod2.setMd5("406022s908ge74sklj");
      compdoc.addExternalModelDefinition(extmod2);

      Submodel submod1 = mplugin.createSubmodel();
      submod1.setId("submod1");
      submod1.setModelRef("Mod1");
      Deletion del1 = submod1.createDeletion();
      del1.setId("deletionA");
      del1.setIdRef("param2");

      Submodel submod2 = new Submodel();
      submod2.setId("submod2");
      submod2.setModelRef("ExtMod1");
      submod2.setSubstanceConversionFactor("subcf");
      submod2.setTimeConversionFactor("tcf");
      submod2.setExtentConversionFactor("xtf");
      Deletion del2 = new Deletion();
      del2.setId("deletionB");
      del2.setMetaIdRef("_0010110");
      rv = submod2.addDeletion(del2);
      del2.setId("deletionC");
      del2.unsetMetaIdRef();
      del2.setPortRef("port2");
      rv = submod2.addDeletion(del2);
      del2.unsetId();
      del2.unsetPortRef();
      del2.setUnitRef("mph");
      rv = submod2.addDeletion(del2);
      del2.unsetUnitRef();
      del2.setIdRef("submodG");
      SBaseRef sbr = del2.createSBaseRef();
      sbr.setIdRef("element5");
      rv = submod2.addDeletion(del2);
      Deletion del3 = new Deletion();
      del3.setIdRef("submodG");
      SBaseRef sbr2 = new SBaseRef();
      sbr2.setIdRef("subsubmodQ");
      SBaseRef subsbr = sbr2.createSBaseRef();
      subsbr.setPortRef("toBdel");
      del3.setSBaseRef(sbr2);
      submod2.addDeletion(del3);
      mplugin.addSubmodel(submod2);

      Port port1 = mplugin.createPort();
      port1.setId("port1");
      port1.setMetaIdRef("_110013");
      Port port2 = new Port();
      port2.setId("port2");
      port2.setIdRef("x");
      mplugin.addPort(port2);
      port2.setId("port3");
      port2.setIdRef("submod2");
      port2.setSBaseRef(sbr2);
      mplugin.addPort(port2);

      libsbml.writeSBMLToFile(document, "comp_example1.xml");
      document = libsbml.readSBMLFromFile("comp_example1.xml");
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
          OStringStream stream = new OStringStream();
          document.printErrors(stream);
          Console.WriteLine("Errors encoutered when round-tripping  SBML file: \n" +
                            stream.str());
          retval = -1;
        }
        libsbml.writeSBMLToFile(document, "comp_example1_rt.xml");
      }

      return retval;
    }
  }
}
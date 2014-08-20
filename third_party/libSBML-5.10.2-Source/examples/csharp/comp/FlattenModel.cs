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

    public class FlattenModel
    {
        public static void Main(String[] args)
        {
            if (args.Length < 2 || args.Length > 3)
            {
                Console.WriteLine("Usage: FlattenModel [-p] input-filename output-filename");
                Console.WriteLine(" -p : list unused ports");
                Environment.Exit(2);
            }

            SBMLReader reader = new SBMLReader();
            SBMLWriter writer = new SBMLWriter();
            bool leavePorts = false;

            SBMLDocument doc;
            if (args.Length == 2)
            {
                doc = reader.readSBML(args[0]);
            }
            else
            {
                doc = reader.readSBML(args[1]);
                leavePorts = true;
            }


            if (doc.getErrorLog().getNumFailsWithSeverity(libsbml.LIBSBML_SEV_ERROR) > 0)
            {
                doc.printErrors();
            }
            else
            {
                /* create a new conversion properties structure */
                ConversionProperties props = new ConversionProperties();

                /* add an option that we want to flatten */
                props.addOption("flatten comp", true, "flatten comp");

                /* add an option to leave ports if the user has requested this */
                props.addOption("leavePorts", leavePorts, "unused ports should be listed in the flattened model");

                /* perform the conversion */
                int result = doc.convert(props);
                if (result != libsbml.LIBSBML_OPERATION_SUCCESS)
                {
                    Console.WriteLine("conversion failed ... ({0})", result);
                    doc.printErrors();
                    Environment.Exit(3);
                }

                if (args.Length == 2)
                {
                    writer.writeSBML(doc, args[1]);
                }
                else
                {
                    writer.writeSBML(doc, args[2]);
                }
            }

        }
    }
}

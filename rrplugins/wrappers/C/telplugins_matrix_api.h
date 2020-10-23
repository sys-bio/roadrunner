/**
 * @file telpluginsmatrix_api.h
 * @brief Header containing tellurium matrix data wrappers
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <hr>
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/

#ifndef telplugins_matrix_apiH
#define telplugins_matrix_apiH
#include "telplugins_exporter.h"
#include "telplugins_types.h"
//---------------------------------------------------------------------------
using namespace tlpc;

#if defined(__cplusplus)
extern "C" {
#endif

/*!
 \brief Returns a pointer to a mtrix underlying data array
 \param matrixH A Handle o a Tellurium matrix type variable
 \return Returns a pointer to a double data array if succesfull, NULL otherwise;
 \ingroup matrix
*/
TLP_C_DS double* tlp_cc tpGetDataArray(TELHandle matrixH);

/*!
 \brief Retrieves the number of rows in a Tellurium matrix object.

 \param data A handle to a Tellurium matrix type variable
 \return Returns the number of rows in the underlying data object.
 \ingroup matrix
*/
TLP_C_DS int tlp_cc tpGetMatrixNumRows(TELHandle data);

/*!
 \brief Retrieves the number of cols in a Tellurium matrix object.

 \param data A handle to a Tellurium matrix type variable
 \return Returns the number of cols in the underlying data object.
 \ingroup matrix
*/
TLP_C_DS int tlp_cc tpGetMatrixNumCols(TELHandle data);

/*!
 \brief Retrieves an element at a given row and column from a Tellurium matrix type variable

 RoadRunner numerical data are indexed from zero

 Example: \code status = getTelluriumDataElement (rrDataHandle, 2, 4, *value); \endcode

 \param matrixH A Handle o a Tellurium matrix type variable
 \param r The row index to the matrixH data
 \param c The column index to the matrixH data
 \param[out] value - The retrieved value from the matrixH data
 \return Returns true if succesful;
 \ingroup matrix
*/
TLP_C_DS bool tlp_cc tpGetMatrixElement(TELHandle matrixH, int r, int c, double *value);

/*!
 \brief Set an data element at a given row and column in a Tellurium matrix type variable

 RoadRunner numerical data are indexed from zero

 Example: \code status = setMatrixElement (matrixHandle, 2, 4, 42); \endcode

 \param matrixH A Handle o a Tellurium matrix type variable
 \param r The row index to the matrixH data
 \param c The column index to the matrixH data
 \param[in] value - The new value.
 \return Returns true if succesful
 \ingroup matrix
*/
TLP_C_DS bool tlp_cc tpSetMatrixElement(TELHandle matrixH, int r, int c, double value);

/*!
 \brief Creates a Tellurium matrix object, and returns a handle to it.

 \param rows Number of rows to create.
 \param cols Number of cols to create.
 \param colNames Column header as a string, e.g. "time, S1, S2".
 \return Returns a handle to a Tellurium matrix object, NULL if unsuccessfull.
 \ingroup matrix
*/
TLP_C_DS TELHandle tlp_cc tpCreateMatrix(int rows, int cols, char* colNames);

/*!
 \brief Free the memory allocated by a Tellurium matrix object.
 \param handle A TELHandle handle to an underlying Matrix object.

 \return Returns a boolean indicating success.
 \ingroup matrix
*/
TLP_C_DS bool tlp_cc tpFreeMatrix(TELHandle handle);


#if defined(__cplusplus)
}    //rrp namespace and extern "C"
#endif

#endif

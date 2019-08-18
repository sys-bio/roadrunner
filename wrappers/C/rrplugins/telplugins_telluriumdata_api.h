/**
 * @file telplugins_roadrunnerdata_api.h
 * @brief Header containing roadrunner data wrappers
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <hr>
 * This file is part of cTellurium.
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

#ifndef telplugins_telluriumdata_apiH
#define telplugins_telluriumdata_apiH
#include "telplugins_exporter.h"
#include "telplugins_types.h"
//---------------------------------------------------------------------------


#if defined(__cplusplus)
using namespace tlpc;
extern "C" {
#endif

/*!
 \brief Retrieve a handle to RoadRunners internal data
 \param handle Handle to a RoadRunner instance
 \return Returns an handle to roadrunners internal data object
 \ingroup roadrunnerdata
*/
TLP_C_DS TELHandle tlp_cc tpGetRoadRunnerDataHandle(TELHandle handle);

/*!
 \brief Retrieves an element at a given row and column from a Tellurium data type variable

 Tellurium numerical data are indexed from zero

 Example: \code status = getTelluriumDataElement (rrDataHandle, 2, 4, *value); \endcode

 \param rrData A Handle o a Tellurium data type variable
 \param r The row index to the rrData data
 \param c The column index to the rrData data
 \param[out] value - The retrieved value from the rrData data
 \return Returns true if succesful;
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpGetTelluriumDataElement(TELHandle rrData, int r, int c, double *value);

/*!
 \brief Set an data element at a given row and column in a Tellurium data type variable

 Tellurium numerical data are indexed from zero

 Example: \code status = setTelluriumDataElement (rrDataHandle, 2, 4, 42); \endcode

 \param rrData A Handle o a Tellurium data type variable
 \param r The row index to the rrData data
 \param c The column index to the rrData data
 \param[in] value - The new value.
 \return Returns true if succesful
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpSetTelluriumDataElement(TELHandle rrData, int r, int c, double value);

/*!
 \brief Retrieves the weight for a data element at a given row and column from a Tellurium data type variable

 Tellurium numerical data are indexed from zero

 Example: \code status = getTelluriumDataWeight (rrDataHandle, 2, 4, *value); \endcode

 \param rrData A Handle o a Tellurium data type variable
 \param r The row index to the rrData data
 \param c The column index to the rrData data
 \param[out] value - The retrieved weight value from the rrData data
 \return Returns true if succesful;
 \note In order to use weights with a roadrunner data object, make sure that data weights are allocated, using the function haveWeights(TELHandle), and
 allocateWeights(rrDataHandle).
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpGetTelluriumDataWeight(TELHandle rrData, int r, int c, double *value);

/*!
 \brief Query if a TelluriumData object have enabled its weights feature
 \param rrData A handle to TelluriumData
 \param answer Boolean indicating if the data has weights or not
 \return Returns true or false indicating function success
 \note Related functions allocateWeights(), getTelluriumDataWeight(), setTelluriumDataWeight()
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpHasWeights(TELHandle rrData, bool* answer);

/*!
 \brief Allocate roadrunner data weights. Initially, all weights are set to '1'
 \param rrData A handle to TelluriumData
 \param success Boolean indicating if allocation was succesful or not
 \return Returns true or false indicating function success
 \note Related functions hasWeights(), getTelluriumDataWeight(), setTelluriumDataWeight()
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpAllocateWeights(TELHandle rrData, bool* success);

/*!
 \brief Set the weight for a Tellurium data element at a given row and column in a Tellurium data type variable

 Tellurium numerical data are indexed from zero

 Example: \code status = setTelluriumDataWeight (rrDataHandle, 2, 4, .89); \endcode

 \param rrData A Handle o a Tellurium data type variable
 \param r The row index to the rrData data
 \param c The column index to the rrData data
 \param[in] value - The new value.
 \return Returns true if succesful
 \note In order to use weights with a roadrunner data object, make sure that data weights are allocated, using the function haveWeights(TELHandle), and
 allocateWeights(rrDataHandle).
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpSetTelluriumDataWeight(TELHandle rrData, int r, int c, double value);

/*!
 \brief Retrieves a Tellurium data column header. The column header is a string
 with comma delimited values. Each value is a header for correseponding data column.

 \param data A handle to a Tellurium data type variable
 \return Returns a string containing the header as a string, NULL if unsuccesful
 \ingroup roadrunnerdata
*/
TLP_C_DS char* tlp_cc tpGetTelluriumDataColumnHeader(TELHandle data);

/*!
 \brief Retrieves a Tellurium data column name by index.
 \param data A handle to a Tellurium data type variable
 \param index Index of requested column name
 \return Returns a string containing the col name as a string, NULL if unsuccesful
 \ingroup roadrunnerdata
*/
TLP_C_DS char* tlp_cc tpGetTelluriumDataColumnHeaderByIndex(TELHandle data, int index);

/*!
 \brief Assign Tellurium data column header from a string.
 The header string is composed of comma delimited (with optional spaces) values. Each value is a header for correseponding data column.

 \param data A handle to a Tellurium data type variable
 \param hdr A Cstring containing the column labels
 \return Returns a boolean indicating if assignement was succesful or not
 \note The number of labels in the string MUST correspond to the number of columns in the data. If not, the assignment is aborted and
 false is returned
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpSetTelluriumDataColumnHeader(TELHandle data, char* hdr);

/*!
 \brief Assign Tellurium data column header from a string for specifiex index.
 \param data A handle to a Tellurium data type variable
 \param index Index for wanted colName to change
 \param hdr A Cstring containing the column label
  \return Returns a boolean indicating if assignement was succesful or not
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpSetTelluriumDataColumnHeaderByIndex(TELHandle data, int index, char* hdr);


/*!
 \brief Retrieves the number of rows in a Tellurium data object.

 \param data A handle to a Tellurium data type variable
 \return Returns the number of rows in the underlying data object.
 \ingroup roadrunnerdata
*/
TLP_C_DS int tlp_cc tpGetTelluriumDataNumRows(TELHandle data);

/*!
 \brief Retrieves the number of cols in a Tellurium data object.

 \param data A handle to a Tellurium data type variable
 \return Returns the number of cols in the underlying data object.
 \ingroup roadrunnerdata
*/
TLP_C_DS int tlp_cc tpGetTelluriumDataNumCols(TELHandle data);

/*!
 \brief Creates a Tellurium data object, and returns a handle to it.

 \param rows Number of rows to create.
 \param cols Number of cols to create.
 \param colNames Column header as a string, e.g. "time, S1, S2".
 \return Returns a handle to a Tellurium data object, NULL if unsuccessfull.
 \ingroup roadrunnerdata
*/
TLP_C_DS TELHandle tlp_cc tpCreateTelluriumData(int rows, int cols, char* colNames);

/*!
 \brief Creates a Tellurium data object from a RoadRunner Data object.

 \param rrData  Handle to a roadrunner data object
 \return Returns a handle to a Tellurium data object, NULL if unsuccessfull.
 \ingroup roadrunnerdata
*/
TLP_C_DS TELHandle tlp_cc tpCreateTelluriumDataFromRoadRunnerData(TELHandle rrData);

/*!
 \brief Free the memory allocated by a Tellurium data object.
 \param handle A TELHandle handle to an underlying TelluriumData object.

 \return Returns a boolean indicating success.
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpFreeTelluriumData(TELHandle handle);

/*!
 \brief Write roadrunner data to file a Tellurium data object.
 \param rrData A handle to Tellurium data.
 \param fName Output file name as a string.
 \return Returns a boolean indicating success.
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpWriteTelluriumDataToFile(TELHandle rrData, const char* fName);

/*!
 \brief Read roadrunner data from a file, into a Tellurium data object.
 \param rrData A handle to Tellurium data.
 \param fName File name to read data from.
 \return Returns a boolean indicating success.
 \ingroup roadrunnerdata
*/
TLP_C_DS bool tlp_cc tpReadTelluriumDataFromFile(TELHandle rrData, const char* fName);


#if defined(__cplusplus)
} //extern "C"
#endif

#endif

/**
 * @file tel_logging_api.h
 * @brief roadRunner C API 2012
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
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

#ifndef telplugins_logging_apiH
#define telplugins_logging_apiH
#include "telplugins_exporter.h"
#include "telplugins_types.h"
//---------------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif

/*!
 \brief RRPLOG level enum
*/
enum  CLogLevel
    {
        CL_PRIO_CURRENT = 0,    /// Use the current level -- don't change the level from what it is.
        CL_PRIO_FATAL,          /// A fatal error. The application will most likely terminate. This is the highest priority.
        CL_PRIO_CRITICAL,       /// A critical error. The application might not be able to continue running successfully.
        CL_PRIO_ERROR,          /// An error. An operation did not complete successfully, but the application as a whole is not affected.
        CL_PRIO_WARNING,        /// A warning. An operation completed with an unexpected result.
        CL_PRIO_NOTICE,         /// A notice, which is an information with just a higher priority.
        CL_PRIO_INFORMATION,    /// An informational message, usually denoting the successful completion of an operation.
        CL_PRIO_DEBUG,          /// A debugging message.
        CL_PRIO_TRACE           /// A tracing message. This is the lowest priority.
    };

enum
{
    clError     = CL_PRIO_ERROR,
    clInfo      = CL_PRIO_INFORMATION,
    clDebug     = CL_PRIO_DEBUG,
    clWarning   = CL_PRIO_WARNING
};

/*!
 \brief Enable logging to console
 \return Returns true if succesful, false otherwise
 \ingroup logging
*/
TLP_C_DS bool tlp_cc tpEnableLoggingToConsole(void);

/*!
 \brief Disable logging to console
 \return Returns true if succesful, false otherwise
 \ingroup logging
*/
TLP_C_DS bool tlp_cc tpDisableLoggingToConsole(void);

/*!
 \brief Enable logging to logFile
 \return Returns true if succesful, false otherwise
 \ingroup logging
*/
TLP_C_DS bool tlp_cc tpEnableLoggingToFile(char* fName);

/*!
 \brief Disable logging to logFile
 \return Returns true if succesful, false otherwise
 \ingroup logging
*/
TLP_C_DS bool tlp_cc tpDisableLoggingToFile();

/*!
 \brief Set the logging status level
 The logging level is determined by the following strings

 "ANY", "DEBUG5", "DEBUG4", "DEBUG3", "DEBUG2", "DEBUG1",
 "DEBUG", "INFO", "WARNING", "ERROR"

 Example: \code setLogLevel ("DEBUG4") \endcode

 Wiil show log messages with levels DEBUG4, DEBUG3 -> Error

 \param lvl Pointer to the logging level string.
 \return Returns true if succesful
 \ingroup logging
*/
TLP_C_DS bool tlp_cc tpSetLogLevel(const char* lvl);

/*!
 \brief Get the logging status level as a pointer to a string

 The logging level can be one of the following strings

 "ANY", "DEBUG5", "DEBUG4", "DEBUG3", "DEBUG2", "DEBUG1",
 "DEBUG", "INFO", "WARNING", "ERROR"

 Example: \code str = getLogLevel (void) \endcode

 \return Returns null if it fails else returns a pointer to the logging string
 \ingroup logging
*/
TLP_C_DS char* tlp_cc tpGetLogLevel(void);

/*!
 \brief Get a pointer to the string that holds the logging file name path

  Example: str = getLogFileName (void)

 \return Returns null if it fails otherwise returns the full path to the logging file name
 \ingroup logging
*/
TLP_C_DS char* tlp_cc tpGetLogFileName(void);

/*!
 \brief Create a log message
 \param lvl Loglevel for message
 \param msg RRPLOG message
 \ingroup logging
*/
TLP_C_DS void tlp_cc tpLogMsg(enum CLogLevel lvl, const char* msg);


#if defined(__cplusplus)
}
#endif

#endif


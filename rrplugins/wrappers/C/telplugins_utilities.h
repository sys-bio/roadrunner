/**
 * @file telplugins_utilities.h
 * @brief Plugins API Utilities header
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
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

#ifndef telplugins_utilitiesH
#define telplugins_utilitiesH
#include "telplugins_exporter.h"
#include "telplugins_types.h"
#include "tel_macros.h"
//---------------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif

/*!
 \brief Global parameter holding last error, if any
*/
extern char* gLastError;

/*!
 \brief Global parameter holding C API install folder
*/
extern char* gInstallFolder;

/*!
 \brief Global parameter holding message for an un-allocated API
*/
extern const char*     ALLOCATE_API_ERROR_MSG;

/*!
 \brief Global parameter holding message when an invalid is handed to a function
*/
extern const char*     INVALID_HANDLE_ERROR_MSG;

TLP_C_DS char* tpCreateText(const char* str);


#if defined(__cplusplus)
}
#endif

#endif

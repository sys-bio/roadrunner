/**
 * @file telplugins_types.h
 * @brief Plugins API Types Header
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib/ for more details.
 *
 * Copyright (C) 2012
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

#ifndef telplugins_typesH
#define telplugins_typesH
#include "telplugins_settings.h"

#if defined(__cplusplus)
namespace tlpc { extern "C"  {
#endif

/*!@brief Void pointer to a RoadRunner instance */
typedef void* TELHandle; /*! Void pointer to a Tellurium object*/

/*!
 \brief Typedef for plugin event function
 \ingroup pluginRoutines
*/
typedef void (event_cc *PluginEvent)(void*, void*);

/*!@brief Enum distinguishing job types */
typedef enum {jtLoadModel, jtSimulate}  RRJobType;


/*!@struct*/
/*!@brief Structure for a set of RoadRunner handles */
typedef struct RRInstanceList
{
    int             Count;                  /*!< The number of elements in the list */
    TELHandle*       Handle;                   /*!< Pointer to a RoadRunner instance */
    void*           RRList;                 /*!< Pointer to a RoadRunnerList instance */
} *RRInstanceListPtr;                      /*!< Pointer to RRInstanceList */

/*!@struct*/
/*!@brief Convenient structure for storing the header and main body source for the generate simulation C code */
typedef struct RRCCode
{
    char*   Header;  /*!< Header file *.h */
    char*   Source;  /*!< Main source code, *.c */

} *RRCCodePtr;    /*!< Pointer to RRCCodePtr struct */




#if defined( __cplusplus)
} }//Namespace
#endif

#endif

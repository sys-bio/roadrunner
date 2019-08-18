/**
 * @file telplugins_cpp_support.h
 * @brief Plugins API CPP Support Functions.
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * 
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
 * redistribute any piece of this software without proper attribution.
 * 
*/

#ifndef telplugins_cpp_supportH
#define telplugins_cpp_supportH
#include <string>
#include <typeinfo>
#include "telplugins_c_api.h"
#include "rrplugins/common/telTelluriumData.h"
#include "rrplugins/core/telPluginManager.h"
#include "rrplugins/common/telProperties.h"
#include "rrplugins/core/telPlugin.h"
#include "rrplugins/common/telProperty.h"
#include "telAPIHandleManager.h"
namespace rr
{
    class RoadRunner;
}

namespace rr
{
class RoadRunnerList;
}

namespace tlpc
{
using std::string;
using rr::PluginManager;
using rr::Properties;
using rr::PropertyBase;
using rr::Plugin;
using rr::Property;

/*!
 \brief Global Handle Manager
*/
extern APIHandleManager gHM;

/*!
 \brief Set API error
 \param[in] error A string containg the error
 \ingroup cpp_support
*/
void tpSetError(const string& err);

/*!
 \brief Cast a handle. This function throws a InvalidHandle exception if it fails
 \param[in] handle  A TELHandle
 \param[in] type Type to cast to, as obtained from type_id().name().
 \param[in] fnc Function in where the cast is done. Obtained from __FUNC__ macro
 \return TELHandle
 \ingroup cpp_support
*/
template<typename  T>
T* castHandle(TELHandle handle, const char* fnc, bool allowBase = false)
{
    T* ptr = static_cast<T*>(gHM.validate(handle, typeid(T*).name(), fnc));
    return ptr;
}

Plugin* tpRegisterPlugin(Plugin* plugin);

//Use internally
/*!
 \brief Utility function to create a C string from a std::string
 \return Returns a char* on success, NULL otherwise
 \ingroup freeRoutines
*/
char*  tpCreateText(const string& str);


/*!
 \brief Cast a handle to RoadRunner Instance list to a RoadRunnerList pointer, throws if it fails
 \param[in] handle  A handle to a RRInstanceList handle
 \return Pointer to a RoadRunnerList instance
 \ingroup cpp_support
*/
TLP_C_DS rr::RoadRunnerList*           getRRList(TELHandle handle);

}

#endif

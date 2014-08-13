/*
 * CachedModel.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ModelResources.h"

#include <rrLogger.h>
#include "rrOSSpecifics.h"
#include <list>

using rr::Logger;
using rr::getLogger;

namespace rrgpu
{

ModelResources::ModelResources()
{
}

ModelResources::~ModelResources()
{
    Log(Logger::LOG_DEBUG) << __FUNC__;
}

} /* namespace rrgpu */

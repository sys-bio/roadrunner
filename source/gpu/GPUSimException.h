/*
 * GPUSimException.h
 *
 *  Created on: Jun 30, 2013
 *      Author: andy
 */

#ifndef RRGPUSimEXCEPTION_H_
#define RRGPUSimEXCEPTION_H_

#include <stdexcept>
#include "rrLogger.h"
#include "rrOSSpecifics.h"

namespace rrgpu
{

class GPUSimException: public std::runtime_error
{
public:
    explicit GPUSimException(const std::string& what) :
            std::runtime_error(what)
    {
    }

    explicit GPUSimException(const std::string& what, const std::string &where) :
            std::runtime_error(what + ", at " + where)
    {
    }
};

#define throw_gpusim_exception(what) \
        {  \
            Log(rr::Logger::LOG_INFORMATION) << "GPUSimException, what: " \
                << what << ", where: " << __FUNC__; \
                throw GPUSimException(what, __FUNC__); \
        }


}

#endif /* RRGPUSimEXCEPTION_H_ */

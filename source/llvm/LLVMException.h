/*
 * LLVMException.h
 *
 *  Created on: Jun 30, 2013
 *      Author: andy
 */

#ifndef RRLLVMEXCEPTION_H_
#define RRLLVMEXCEPTION_H_

#include <stdexcept>
#include "rrLogger.h"
#include "rrOSSpecifics.h"

namespace rrllvm
{
    /** @cond PRIVATE */

    class LLVMException : public std::runtime_error
    {
    public:
        explicit LLVMException(const std::string& what) :
            std::runtime_error(what)
        {
        }

        explicit LLVMException(const std::string& what, const std::string& where) :
            std::runtime_error(what + ", at " + where)
        {
        }
    };

#define throw_llvm_exception(what) \
        do {  \
            rrLog(rr::Logger::LOG_INFORMATION) << "LLVMException, what: " \
                << what << ", where: " << __FUNC__; \
                throw LLVMException(what, __FUNC__); \
        } while (0)

    /** @endcond PRIVATE */
}

#endif /* RRLLVMEXCEPTION_H_ */

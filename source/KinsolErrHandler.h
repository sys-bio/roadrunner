//
// Created by Ciaran on 27/02/2021.
//

#ifndef ROADRUNNER_KINSOLERRHANDLER_H
#define ROADRUNNER_KINSOLERRHANDLER_H

#include "kinsol/kinsol.h"
#include <iostream>
#include <sstream>

namespace rr {

    /**
     * @brief Convert kinsol error codes into text.
     * @param cvodeError the error code
     * @author CW
     * @details These error messages are taken directly
     * from the kinsol documentation. More information
     * can be found in 9.1.5
     */
    std::string decodeKinsolError(int cvodeError);


    /**
    *
    * @brief Processes error and warning messages from Kinsol. Conforms to the
     * function pointer "KINErrHandlerFn".
    *
    * @param error_code: the error code.
    * @param module: is the name of the kinsol module reporting the error.
    * @param function: is the name of the function in which the error occurred.
    * @param msg: is the error message.
    * passed to CVodeSetErrHandlerFn.
    *
    * Notes:
    * error_code is negative for errors and positive (KIN_WARNING) for warnings.
    * If a function that returns a pointer to memory encounters an error,
    * it sets error_code to 0.
    */
    void kinsolErrHandler(int line, const char* function, const char* file,
        const char* msg, SUNErrCode error_code,
        void* err_user_data, SUNContext sunctx);

    /**
    * macro to throw an error message
    */
//#define handleKinsolError(errCode) \
//    std::string what = decodeKinsolError(errCode); \
//    throw KinsolException(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ":" + std::string(__FUNCTION__) + " " + what);

}

#endif //ROADRUNNER_KINSOLERRHANDLER_H

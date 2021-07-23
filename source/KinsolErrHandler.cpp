//
// Created by Ciaran on 27/02/2021.
//

#include "KinsolErrHandler.h"
#include "kinsol/kinsol.h"
#include "rrLogger.h"
#include <iostream>
#include <sstream>

namespace rr {

    /**
     * @brief Convert kinsol error codes into text.
     * @param kinsolError the error code
     * @details These error messages are taken directly
     * from the kinsol documentation. More information
     * can be found in 9.1.5.
     *
     */
    std::string decodeKinsolError(int errCode) {
        std::ostringstream errMsg;

        char* errorString = KINGetReturnFlagName(errCode);
        errMsg << errorString << ":";

        switch (errCode) {
            case KIN_MEM_NULL: {
                errMsg << ": The kinsol memory block was not initialized through a "
                          "previous call to KINCreate." << std::endl;
                break;
            }
            case KIN_ILL_INPUT: {
                errMsg << ": An input argument to KINInit has an illegal value" << std::endl;
                break;
            }
            case KIN_NO_MALLOC: {
                errMsg << "The kinsol memory was not allocated by a call to KINCreate." << std::endl;
                break;
            }
            case KIN_MEM_FAIL: {
                errMsg << "A memory allocation request has failed." << std::endl;
                break;
            }
            case KIN_LINESEARCH_NONCONV: {
                errMsg << "The line search algorithm was unable to find an iterate suffciently distinct from the "
                          "current iterate, or could not find an iterate satisfying the suffcient decrease condition. "
                          "Failure to satisfy the suffcient decrease condition could mean the current iterate "
                          "is close to an approximate solution of the given nonlinear system, the difference "
                          "approximation of the matrix-std::vector product J(u)v is inaccurate, or the real scalar "
                          "scsteptol is too large." << std::endl;
                break;
            }
            case KIN_MAXITER_REACHED: {
                errMsg << "The maximum number of nonlinear iterations has been reached." << std::endl;
                break;
            }
            case KIN_MXNEWT_5X_EXCEEDED: {
                errMsg << "Five consecutive steps have been taken that satisfy the inequality kDupkL2 > 0:99 "
                          "mxnewtstep, where p denotes the current step and mxnewtstep is a scalar upper "
                          "bound on the scaled step length. Such a failure may mean that kDFF(u)kL2 asymptotes "
                          "from above to a positive value, or the real scalar mxnewtstep is too small. "
                       << std::endl;
                break;
            }
            case KIN_LINESEARCH_BCFAIL: {
                errMsg << "The line search algorithm was unable to satisfy the \\beta-condition\" for MXNBCF +1 "
                          "nonlinear iterations (not necessarily consecutive), which may indicate the algorithm "
                          "is making poor progress." << std::endl;
                break;
            }
            case KIN_LINSOLV_NO_RECOVERY: {
                errMsg
                        << "The user-supplied routine psolve encountered a recoverable error, but the preconditioner "
                           "is already current." << std::endl;
                break;
            }
            case KIN_LINIT_FAIL: {
                errMsg << "The kinls initialization routine (linit) encountered an error." << std::endl;
                break;
            }
            case KIN_LSETUP_FAIL: {
                errMsg << "The kinls setup routine (lsetup) encountered an error; e.g., the user-supplied routine "
                          "pset (used to set up the preconditioner data) encountered an unrecoverable error. It is possible "
                          "that the jacobian for your system is singular. If not already, set conservedMoietyAnalysis = true"
                       << std::endl;
                break;
            }
            case KIN_LSOLVE_FAIL: {
                errMsg << "The kinls solve routine (lsolve) encountered an error; e.g., the user-supplied routine "
                          "psolve (used to to solve the preconditioned linear system) encountered an unrecoverable "
                          "error." << std::endl;
                break;
            }
            case KIN_SYSFUNC_FAIL: {
                errMsg << "The system function failed in an unrecoverable manner." << std::endl;
                break;
            }
            case KIN_FIRST_SYSFUNC_ERR: {
                errMsg << "The system function failed recoverably at the first call." << std::endl;
                break;
            }
            case KIN_REPTD_SYSFUNC_ERR: {
                errMsg << "The system function had repeated recoverable errors. No recovery is possible."
                       << std::endl;
                break;
            }
            case KIN_VECTOROP_ERR: {
                errMsg << "a std::vector operation error occurred" << std::endl;
                break;
            }
            default:
                throw std::runtime_error("Unrecognized kinsol errorunhandled exception");
        }
        free(errorString);
        return errMsg.str();
    }


    /**
    *
    * @brief Processes error and warning messages from Kinsol. Conforms to the
     * function pointer "KINErrHandlerFn".
    *
    * @param error_code: the error code.
    * @param module: is the name of the kinsol module reporting the error.
    * @param function: is the name of the function in which the error occurred.
    * @param msg: is the error message.
    * @param eh_data: is a pointer to user data, the same as the eh_data parameter
    * passed to CVodeSetErrHandlerFn.
    *
    * Notes:
    * error_code is negative for errors and positive (KIN_WARNING) for warnings.
    * If a function that returns a pointer to memory encounters an error,
    * it sets error_code to 0.
    */
    void kinsolErrHandler(int error_code, const char *module, const char *function, char *msg) {
        // note, we do not use msg. The error messages in decodeKinsolError are from the docs
        // and are more detailed than those that kinsol automatically provides with msg.
        if (error_code < 0) {
            std::ostringstream err;
            err << "Kinsol Error: Module: " << module << ", Function: " << function
                                     << "Message: " << decodeKinsolError(error_code);
            throw std::runtime_error(err.str());
        } else if (error_code > 0) {
            rrLog(Logger::LOG_WARNING) << "Kinsol Warning: "
                                       << ", Module: " << module << ", Function: " << function
                                       << ", Message: " << msg;
        }
    }

}


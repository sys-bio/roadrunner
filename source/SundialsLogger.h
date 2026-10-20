#ifndef RR_SUNDIALSLOGGER_H_
#define RR_SUNDIALSLOGGER_H_

#include <sundials/sundials_context.h>

namespace rr {

/**
 * @brief Sends the messages that SUNDIALS writes to a SUNContext's SUNLogger to roadrunner's Logger.
 * @details SUNDIALS 7 writes the warnings of its modules (for example CVODE's
 * "t + h = t on the next step") to the SUNLogger of the SUNContext, which by default
 * prints them to stdout. This installs functions on that SUNLogger that give each
 * message to rr::Logger instead, at the corresponding level (warning, information,
 * debug), so that the messages follow the Logger level, format, file and console
 * stream, and no longer appear on stdout.
 *
 * SUNDIALS errors are not forwarded: each SUNContext roadrunner creates already has an
 * error handler that logs them (with the decoded error code), and forwarding them as well
 * would log every error twice.
 *
 * Needs SUNDIALS 7.8.0 or later (SUNLogger_SetQueueAndFlushMsgFns).
 *
 * @param sunctx a valid SUNContext. The functions stay in use until it is freed.
 */
void routeSundialsLoggingToRoadRunnerLogger(SUNContext sunctx);

} // namespace rr

#endif // RR_SUNDIALSLOGGER_H_

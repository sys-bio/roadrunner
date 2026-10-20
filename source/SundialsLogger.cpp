#include "SundialsLogger.h"

#include "rrLogger.h"

#include <sundials/sundials_config.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_logger.h>

#include <string>

#if SUNDIALS_VERSION_MAJOR < 7 || (SUNDIALS_VERSION_MAJOR == 7 && SUNDIALS_VERSION_MINOR < 8)
#error "roadrunner needs SUNDIALS 7.8.0 or later: it uses SUNLogger_SetQueueAndFlushMsgFns"
#endif

namespace rr {

    /**
     * Called by SUNDIALS for each message that reaches the SUNLogger, with the level, the
     * file and line (scope) and the function (label) that issued it, and the formatted text.
     */
    static SUNErrCode queueSundialsMessage(SUNLogger /*logger*/, SUNLogLevel lvl, const char* /*prefix*/,
                                           int /*rank*/, const char* scope, const char* label,
                                           const char* payload, void* /*content*/) {
        Logger::Level level;
        switch (lvl) {
            case SUN_LOGLEVEL_WARNING:
                level = Logger::LOG_WARNING;
                break;
            case SUN_LOGLEVEL_INFO:
                level = Logger::LOG_INFORMATION;
                break;
            case SUN_LOGLEVEL_DEBUG:
                level = Logger::LOG_DEBUG;
                break;
            default:
                // Errors are logged by the error handlers of the SUNContexts (see the header).
                return SUN_SUCCESS;
        }

        // Skip building the message when the Logger would discard it.
        if (Logger::getLevel() < level) {
            return SUN_SUCCESS;
        }

        std::string text = payload ? payload : "";
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }

        // for example "CVode (cvodes.c:3528): Internal t = 613190 and h = ... are such that t + h = t ..."
        std::string message = label ? label : "SUNDIALS";
        if (scope && *scope) {
            message += std::string(" (") + scope + ")";
        }
        message += ": " + text;

        Logger::log(level, message);
        return SUN_SUCCESS;
    }

    // The Logger writes each message when it is logged, so there is nothing to flush.
    static SUNErrCode flushSundialsMessages(SUNLogger /*logger*/, SUNLogLevel /*lvl*/, void* /*content*/) {
        return SUN_SUCCESS;
    }

    void routeSundialsLoggingToRoadRunnerLogger(SUNContext sunctx) {
        SUNLogger logger = nullptr;
        if (SUNContext_GetLogger(sunctx, &logger) != SUN_SUCCESS || logger == nullptr) {
            rrLog(Logger::LOG_WARNING) << "Could not get the SUNLogger of a SUNContext, so SUNDIALS messages "
                                          "will not appear in the roadrunner log.";
            return;
        }
        if (SUNLogger_SetQueueAndFlushMsgFns(logger, queueSundialsMessage, flushSundialsMessages, nullptr)
            != SUN_SUCCESS) {
            rrLog(Logger::LOG_WARNING) << "Could not route the SUNDIALS messages to the roadrunner log.";
        }
    }

} // namespace rr

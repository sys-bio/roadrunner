#!/usr/bin/env python

import roadrunner as rr

rr.Logger.setLevel(rr.Logger.LOG_TRACE)

def log():
    rr.Logger.log(rr.Logger.LOG_FATAL, "A fatal message")
    rr.Logger.log(rr.Logger.LOG_CRITICAL, "A critical message")
    rr.Logger.log(rr.Logger.LOG_ERROR, "An error message")
    rr.Logger.log(rr.Logger.LOG_WARNING, "A warning message")
    rr.Logger.log(rr.Logger.LOG_NOTICE, "A notice message")
    rr.Logger.log(rr.Logger.LOG_INFORMATION, "An informational message")
    rr.Logger.log(rr.Logger.LOG_DEBUG, "A debugging message.")
    rr.Logger.log(rr.Logger.LOG_TRACE,  "A tracing message. This is the lowest priority.")

rr.Logger.log(rr.Logger.LOG_NOTICE, "logging with default colors:")
log()

rr.Logger.setFormattingPattern("%Y-%m-%d %H:%M:%S %p: %t")
log()


rr.Logger.setProperty("traceColor", "red")
rr.Logger.setProperty("debugColor", "green")
rr.Logger.setProperty("informationColor", "blue")
rr.Logger.setProperty("noticeColor", "magenta")
rr.Logger.setProperty("warningColor", "cyan")
rr.Logger.setProperty("errorColor", "gray")
rr.Logger.setProperty("criticalColor", "white")
rr.Logger.setProperty("fatalColor", "brown")

rr.Logger.log(rr.Logger.LOG_NOTICE, "logging with new and better colors:")
log()



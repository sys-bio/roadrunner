macro(write_git_info)

    # pre-process the GitInfo.h.in file and insert the response from git about
    # last commit and current branch.
    # may not have git, set default replacement strings then.
    find_program(GIT git)

    if (GIT)
        message("found git: ${GIT}")
        execute_process(
                COMMAND ${GIT} symbolic-ref -q --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE ROADRUNNER_GIT_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )


        execute_process(
                COMMAND ${GIT} log -1 --format=%H
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE ROADRUNNER_GIT_LAST_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else ()
        # could not find GIT
        message("could not find git")

        set(ROADRUNNER_GIT_BRANCH "git not found, unknown git branch")
        set(ROADRUNNER_GIT_LAST_COMMIT "git not found, unknown git commit")
    endif (GIT)

    # writes the results of the git commands to GitInfo.h
    message(STATUS "ROADRUNNER_GIT_BRANCH        : ${ROADRUNNER_GIT_BRANCH}")
    message(STATUS "ROADRUNNER_GIT_LAST_COMMIT   : ${ROADRUNNER_GIT_LAST_COMMIT}")

endmacro()
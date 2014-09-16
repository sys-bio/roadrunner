// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * CudaCodeCompiler.cpp
 *
 *  Created on: 09/16/2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "CudaCodeCompiler.hpp"
# include "rrUtils.h"

#include "Poco/SharedLibrary.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
// #include <sys/time.h>
#include <fcntl.h>
#include <time.h>

// == CODE ====================================================

#define USE_POPEN 0

namespace rr
{

namespace rrgpu
{

    class CudaExecutableModuleImpl {
    public:
        typedef CudaExecutableModule::EntryPointSig EntryPointSig;

        CudaExecutableModuleImpl() {

        }

        CudaExecutableModuleImpl(const CudaExecutableModuleImpl& other)
          : libname_(other.libname_), entryMangled_(other.entryMangled_) {
            load();
        }

        EntryPointSig getEntry() const { return entry_; }

    protected:
        friend class CudaCodeCompiler;

        // loads the library specified by libname_
        void load() {
            auto load_start = std::chrono::high_resolution_clock::now();

            // load the module

            try {
                so_.load(libname_);
            } catch(Poco::LibraryLoadException e) {
                throw_gpusim_exception("Cannot load lib: " + e.message());
            }

            Log(Logger::LOG_DEBUG) << "Loading symbol " << entryMangled_ << " in " << libname_;
            if(!so_.hasSymbol(entryMangled_))
                throw_gpusim_exception("Lib " + libname_ + " has no symbol \"" + entryMangled_ + "\"");

            Log(Logger::LOG_TRACE) << "Entering CUDA code";

            entry_ = (EntryPointSig)so_.getSymbol(entryMangled_);

            auto load_finish = std::chrono::high_resolution_clock::now();

            Log(Logger::LOG_INFORMATION) << "Loading model took " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish - load_start).count() << " ms";
        }

        std::string libname_;
        std::string entryMangled_;
        Poco::SharedLibrary so_;
        EntryPointSig entry_ = nullptr;
    };

    CudaExecutableModule::CudaExecutableModule()
      : impl_(new CudaExecutableModuleImpl()) {}

    CudaExecutableModule::CudaExecutableModule(const CudaExecutableModule& other)
      : impl_(other.impl_ ? new CudaExecutableModuleImpl(*other.impl_) : new CudaExecutableModuleImpl()) {}

    CudaExecutableModule::~CudaExecutableModule() {}

    CudaExecutableModule::EntryPointSig CudaExecutableModule::getEntry() const {
        return impl_->getEntry();
    }

    using namespace dom;

    CudaExecutableModule CudaCodeCompiler::generate(CudaModule& mod, const std::string hashedid, const std::string& entryName) {
        CudaExecutableModule result;

        std::string cuda_src_name = joinPath(getTempDir(), "rr_cuda_model_" + hashedid + ".cu");

        Log(Logger::LOG_DEBUG) << "CUDA source: " << cuda_src_name;

        result.impl_->libname_ = joinPath(getTempDir(), "rr_cuda_model_" + hashedid + ".so");

        Log(Logger::LOG_DEBUG) << "CUDA object: " << result.impl_->libname_;

        auto serialize_start = std::chrono::high_resolution_clock::now();

        // serialize the module to a document

        {
            Serializer s(cuda_src_name);
            mod.serialize(s);
        }

        auto serialize_finish = std::chrono::high_resolution_clock::now();

        Log(Logger::LOG_INFORMATION) << "Serializing model took " << std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish - serialize_start).count() << " ms";

        auto compile_start = std::chrono::high_resolution_clock::now();

        // compile the module

        std::string popenline = "nvcc -D__CUDACC__ -ccbin gcc -m32 --ptxas-options=-v --compiler-options '-fPIC' -Drr_cuda_model_EXPORTS -Xcompiler ,\"-fPIC\",\"-fPIC\",\"-g\" -DNVCC --shared -o "
        + result.impl_->libname_ + " "
        + cuda_src_name + " 2>&1 >/dev/null";

        Log(Logger::LOG_DEBUG) << "CUDA compiler line: " << popenline;

        #define SBUFLEN 4096
        char sbuf[SBUFLEN];

        FILE* pp = NULL;
# if USE_POPEN
        pp = popen(popenline.c_str(), "r");

        fgets(sbuf, SBUFLEN, pp);

        // 512 for warning
        int code = pclose(pp)/256;
        pp = NULL;
# else
        int filedes[2];
        ssize_t nbytes;

        // http://www.microhowto.info/howto/capture_the_output_of_a_child_process_in_c.html

        if (int pipecode = pipe(filedes) == -1)
            throw_gpusim_exception("Failed to pipe");

        int pid = fork();
        switch (pid) {
            case -1:
                throw_gpusim_exception("Failed to fork");
            case 0:
                // Child - writes to pipe
                // close read descriptor
                close(filedes[0]);
                // route stdout to the pipe
//                 while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
                // route stderr to the pipe
                while ((dup2(filedes[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}

                close(filedes[1]);

                execl("/usr/local/cuda-6.0/bin/nvcc", "nvcc", "-D__CUDACC__", "-ccbin", "gcc", "-m32", "--ptxas-options=-v", "--compiler-options", "'-fPIC'", "-Drr_cuda_model_EXPORTS", "-Xcompiler", ",\"-fPIC\",\"-fPIC\",\"-g\"", "-DNVCC", "--shared", "-o", result.impl_->libname_.c_str(), cuda_src_name.c_str(), (char*)0);

//                 execl("/bin/ls", "ls", "/home/jkm", (char*)0);

                perror("execl");
                _exit(1);
            default:
                // Parent - reads from pipe
                close(filedes[1]);

//                 timeval tinterval;
//                 fd_set fdset;
//                 for (int z=0; z<10; ++z) {
//                     std::cerr << "Waiting" << std::endl;
//                     tinterval.tv_sec = 1;
//                     tinterval.tv_usec = 0;
//                     FD_ZERO(&fdset);
//                     FD_SET(filedes[0], &fdset);
//                     int selcode;
//                     if ((selcode = select(1, &fdset, NULL, NULL, &tinterval)) == -1) {
//                         throw_gpusim_exception("Problem with pipe");
//                     } else if (selcode) {
// //                     if (FD_ISSET(filedes[0], &fdset)) {
//                         std::cerr << "Ready" << std::endl;
//                         break;
//                     }
//                     std::cerr << "selcode = " << selcode << std::endl;
//                 }

                int flags = fcntl(filedes[0], F_GETFL, 0);
                if (fcntl(filedes[0], F_SETFL, O_NONBLOCK | flags) == -1)
                    throw_gpusim_exception("Problem setting flags");
//                 std::cerr << "flags pre " << std::hex << flags << " flags post " << std::hex << fcntl(filedes[0], F_GETFL, 0) << " O_NONBLOCK " << O_NONBLOCK << std::endl;
                timespec tspec;
                int c=0;
                std::string throbber_semi[4] = {"◐", "◓", "◑", "◒"};
                if (Logger::getLevel() >= Logger::LOG_NOTICE)
                    std::cerr << "Compiling CUDA code..." <<  throbber_semi[0];
//                 for (int z=0; z<1000; ++z) {
                while (1) {
//                     std::cerr << "waiting\n";
                    if (++c >= 4)
                        c = 0;
//                     std::cerr << c << std::endl;
//                     std::cerr << throbber_semi[0] << std::endl;
//                     std::cerr << throbber_semi[1] << std::endl;
//                     std::cerr << throbber_semi[2] << std::endl;
//                     std::cerr << throbber_semi[3] << std::endl;
                    if (Logger::getLevel() >= Logger::LOG_NOTICE)
                        std::cerr << "\b" << throbber_semi[c];

                    // sleep
                    tspec.tv_sec = 0;
//                     tspec.tv_nsec = 0;
//                     tspec.tv_nsec = 100000000;
                    tspec.tv_nsec = 10000000;
                    if (nanosleep(&tspec, NULL) < 0)
                        throw_gpusim_exception("Problem with nanosleep");

                    errno = 0;
                    nbytes = read(filedes[0], sbuf, sizeof(sbuf));
                    if (errno != EAGAIN) {
//                         std::cerr << "finished reading";
                        if (Logger::getLevel() >= Logger::LOG_NOTICE)
                            std::cerr << "\bDone\n";
                        break;
                    }
                }
                close(filedes[0]);
        }

        int code = 0;
//         std::cerr << "Wait for child to finish" << std::endl;
        if (waitpid(pid, &code, 0) == -1)
            throw_gpusim_exception("Problem with child process");

# endif
        Log(Logger::LOG_DEBUG) << "nvcc return code: " << code << "\n";

        if(code) {
            std::stringstream ss;
            ss << "nvcc code: " << code << "\n";
            ss << "Compiler errors:\n" << sbuf << "\n";
            throw_gpusim_exception(ss.str());
        }

        // get the mangled name of the entry point

        std::string demangleCmd = "nm -g /tmp/rr_cuda_model.so | grep -ohe '_.*" + entryName + "[^\\s]*$'";
        pp = popen(demangleCmd.c_str(), "r");

        fgets(sbuf, SBUFLEN, pp);
        result.impl_->entryMangled_ = sbuf;

        code = pclose(pp)/256;

        if(code) {
            std::stringstream ss;
            ss << "Could not find symbol: " << entryName << "\n";
            throw_gpusim_exception(ss.str());
        }

        // strip newline

        while(result.impl_->entryMangled_.back() == '\n' && result.impl_->entryMangled_.size())
            result.impl_->entryMangled_ = std::string(result.impl_->entryMangled_,0,result.impl_->entryMangled_.size()-1);

        auto compile_finish = std::chrono::high_resolution_clock::now();

        Log(Logger::LOG_INFORMATION) << "Compiling model took " << std::chrono::duration_cast<std::chrono::milliseconds>(compile_finish - compile_start).count() << " ms";

        result.impl_->load();

        return result;
    }

} // namespace rrgpu

} // namespace rr

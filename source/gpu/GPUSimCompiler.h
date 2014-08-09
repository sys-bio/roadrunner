/*
 * GPUSimCompiler.h
 *
 * Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef GPUSimCompilerH
#define GPUSimCompilerH

#include "rrCompiler.h"

namespace rrgpu
{

class RR_DECLSPEC GPUSimCompiler: public rr::Compiler
{
public:
    GPUSimCompiler();
    virtual ~GPUSimCompiler();

    virtual std::string getCompiler() const;
    virtual bool setCompiler(const std::string& compiler);

    virtual std::string getCompilerLocation() const;
    virtual bool setCompilerLocation(const std::string& path);

    virtual std::string getSupportCodeFolder() const;
    virtual bool setSupportCodeFolder(const std::string& path);


    /**
     * getDefaultTargetTriple() - Return the default target triple the compiler
     * has been configured to produce code for.
     *
     * The target triple is a string in the format of:
     *   CPU_TYPE-VENDOR-OPERATING_SYSTEM
     * or
     *   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM
     */
    virtual std::string getDefaultTargetTriple();

    /**
     * getProcessTriple() - Return an appropriate target triple for generating
     * code to be loaded into the current process, e.g. when using the JIT.
     */
    virtual std::string getProcessTriple();

    /**
     * getHostCPUName - Get the GPUSim name for the host CPU. The particular format
     * of the name is target dependent, and suitable for passing as -mcpu to the
     * target which matches the host.
     *
     * \return - The host CPU name, or empty if the CPU could not be determined.
     */
    virtual std::string getHostCPUName();

    /**
     * get the version of the compiler.
     */
    virtual std::string getVersion();
};

} /* namespace rr */
#endif /* GPUSimCompilerH */

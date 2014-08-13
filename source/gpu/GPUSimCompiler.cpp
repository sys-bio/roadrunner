/*
 * GPUSimCompiler.cpp
 *
 * Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#pragma hdrstop
#include "GPUSimCompiler.h"
#include "rrUtils.h"
#include "GPUSimException.h"
#include <sstream>
#include <ctime>

namespace rr
{

namespace rrgpu
{

GPUSimCompiler::GPUSimCompiler()
{
}

GPUSimCompiler::~GPUSimCompiler()
{
}

std::string GPUSimCompiler::getCompiler() const
{
    return "GPUSim";
}

bool GPUSimCompiler::setCompiler(const std::string& compiler)
{
    return true;
}

std::string GPUSimCompiler::getCompilerLocation() const
{
    return "not used";
}

bool GPUSimCompiler::setCompilerLocation(const std::string& path)
{
    return true;
}

std::string GPUSimCompiler::getSupportCodeFolder() const
{
    return "not used";
}

bool GPUSimCompiler::setSupportCodeFolder(const std::string& path)
{
    return true;
}

std::string GPUSimCompiler::getDefaultTargetTriple()
{
    throw_gpusim_exception("getDefaultTargetTriple() is not supported");
}

std::string GPUSimCompiler::getProcessTriple()
{
    throw_gpusim_exception("getProcessTriple() is not supported");
}

std::string GPUSimCompiler::getHostCPUName()
{
    throw_gpusim_exception("getHostCPUName() is not supported");
}

std::string GPUSimCompiler::getVersion()
{
    std::stringstream ss;
    ss << "0.1";
    return ss.str();
}

} // namespace rrgpu

} // namespace rr

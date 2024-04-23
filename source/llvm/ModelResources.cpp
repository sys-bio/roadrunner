/*
 * CachedModel.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andy
 */
#pragma hdrstop

#include "ModelResources.h"
#include "Random.h"
#include "JitFactory.h"
#include <rrLogger.h>
#include <rrStringUtils.h>
#include "rrRoadRunnerOptions.h"
#include "MCJit.h"
#include "LLJit.h"
#include "rrRoadRunnerOptions.h"

#include <memory>

#undef min
#undef max


#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#pragma warning(disable: 4244)
#endif

#include "llvm/SBMLSupportFunctions.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#pragma warning(default: 4244)
#endif

using rr::Logger;
using rr::getLogger;
using rr::LoadSBMLOptions;

namespace rrllvm {
    /** @cond PRIVATE */

    ModelResources::ModelResources() 
        : symbols(nullptr)
        , executionEngine(nullptr)
        , context(nullptr)
        , random(nullptr)
        , errStr(nullptr)
        , jit(nullptr)
    {
        // the reset of the ivars are assigned by the generator,
        // and in an exception they are not, does not matter as
        // we don't have to delete them.
    }

    ModelResources::~ModelResources() {
        rrLog(Logger::LOG_DEBUG) << __FUNC__;

        if (errStr && !errStr->empty()) {
            rrLog(Logger::LOG_WARNING) << "Non-empty LLVM ExecutionEngine error std::string: " << *errStr;
        }

        delete symbols;

        // the exe engine owns all the functions
//    delete executionEngine;
//    delete context;
        delete random;
    }

    void ModelResources::saveState(std::ostream &out) const {
        symbols->saveState(out);
        rr::saveBinary(out, sbmlMD5);

        /**
         * This bit of code is unintuitive really.
         * The goal is to save the moduleStr variable for loading.
         * Two situations arise:
         *  1) moduleStr is not populated, but can be obtained from
         *     jit->compiledModuleBinaryStream (see MCJit::writeObjectToBinaryStream)
         *  2) moduleStr is populated with the string we need
         * Situation 1 occurs if this model has not been saved/loaded before.
         * Situation 2 occurs if the model we are saving is itself a copy of an
         * original model. This is a bad design but I can't figure out why
         * jit->compiledModuleBinaryStream is empty second time around.
         */
        std::string moduleAsString;
        if (moduleStr.empty()){
            /**
             * If we already have a moduleStr, use it. This happens
             * when a model has been saved and loaded already, like in:
             *     RoadRunner rr(OpenLinearFlux().str());
             *     RoadRunner rr2;
             *     rr2 = rr; // first time: rr.saveState used. rr2.loadState used
             */
            moduleAsString = jit->getModuleAsString(sbmlMD5);
        } else {
            /**
             * If we already have a moduleStr, use it. This happens
             * when a model has been saved and loaded already, like in:
             *     RoadRunner rr(OpenLinearFlux().str());
             *     RoadRunner rr2;
             *     rr2 = rr; // first time: rr.saveState used. rr2.loadState used
             *     rr = rr2; // second time: rr2.saveState and rr.loadState
             */
            moduleAsString = moduleStr;
        }

        rr::saveBinary(out, moduleAsString);

        // MCJit uses this variable. LLJit does not. Both are needed due to
        // awkwardness of having old and new llvm based code.
        bool usingCompiledModuleBinaryStream = false;

        // see comment in loadState
        if (jit->compiledModuleBinaryStream) {
            usingCompiledModuleBinaryStream = true;
        }
        rr::saveBinary(out, usingCompiledModuleBinaryStream);

        if (usingCompiledModuleBinaryStream) {
            rr::saveBinary(out, jit->compiledModuleBinaryStream.get()->str().str());
        }

    }

    void ModelResources::loadState(std::istream &in, uint modelGeneratorOpt) {
        jit.reset(JitFactory::makeJitEngine(modelGeneratorOpt));

        // get rid of sumbols, if not nullptr.
        // todo make symbols a unique_ptr
        delete symbols;
        //load the model data symbols from the stream
        symbols = new LLVMModelDataSymbols(in);

        rr::loadBinary(in, sbmlMD5);
        assert(!sbmlMD5.empty() && "sbml md5 is empty");

        rr::loadBinary(in, moduleStr);

        // flag which determines whether we are using
        // compiledModuleBinaryStream as moduleStr storage (MCJit)
        // or whether this is automatically cached by LLVM (LLJit)
        // meaning we can directly access what we need from the cache.
        bool usingCompiledModuleBinaryStream = false;
        rr::loadBinary(in, usingCompiledModuleBinaryStream);


        /**
         * Although we have a LLVM IR module as binary string (moduleStr)
         * and this is all that is needed to create another model from the
         * input stream, we still need to have a populated
         * jit->compiledModuleBinaryStream variable, otherwise a copy of
         * a roadrunner instance cannot saveState.
         */
        if (usingCompiledModuleBinaryStream) {
            //We saved this as a string; convert it back to a 'raw_svector_ostream'
            string stream_str;
            rr::loadBinary(in, stream_str);
            llvm::raw_svector_ostream* binarystream = new llvm::raw_svector_ostream(jit->moduleBuffer);
            *binarystream << stream_str;

            jit->compiledModuleBinaryStream.reset(binarystream);
        }

        //Set up a buffer to read the object code from
        std::unique_ptr<llvm::MemoryBuffer> memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));

        llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objFile =
                llvm::object::ObjectFile::createObjectFile(
                        llvm::MemoryBufferRef(moduleStr, sbmlMD5));
        if (!objFile) {
            std::string err = "Failed to load object data.";
            rrLogErr << err;
            llvm::logAllUnhandledErrors(objFile.takeError(), llvm::errs(), err);
        }
        llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject(
                std::move(*objFile), std::move(memBuffer));

        jit->addObjectFile(std::move(owningObject));
        jit->mapLLVMGeneratedFunctionsToSymbols(this, modelGeneratorOpt);

    }

    /** @endcond PRIVATE */
} /* namespace rrllvm */

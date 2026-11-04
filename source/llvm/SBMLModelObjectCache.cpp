//
// Created by Ciaran on 05/11/2021.
//

#include "SBMLModelObjectCache.h"
#include "rrLogger.h"
#include <iostream>

using namespace rr;

namespace rrllvm {

    SBMLModelObjectCache &SBMLModelObjectCache::getObjectCache() {
        static SBMLModelObjectCache objectCacheInstance;
        return objectCacheInstance;
    }

    void SBMLModelObjectCache::notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef ObjBuffer) {
        rrLogDebug << "module: " << M->getModuleIdentifier() << " is compiled";
        store(M->getModuleIdentifier(), ObjBuffer);
    }

    std::unique_ptr<llvm::MemoryBuffer> SBMLModelObjectCache::getObject(const llvm::Module *M) {
        ObjectPtr obj = find(M->getModuleIdentifier());
        if (!obj) {
            rrLogDebug << "No object for " << M->getModuleIdentifier() << " in cache. Compiling";
            return nullptr;
        }

        rrLogDebug << "Object for " << M->getModuleIdentifier() << " loaded from cache.";
        // a copy, because the caller may keep it longer than the cache keeps the object
        return llvm::MemoryBuffer::getMemBufferCopy(obj->getBuffer(), obj->getBufferIdentifier());
    }

    SBMLModelObjectCache::ObjectPtr SBMLModelObjectCache::find(const std::string &key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto I = cachedObjects.find(key);
        if (I == cachedObjects.end()) {
            return nullptr;
        }
        return I->second;
    }

    SBMLModelObjectCache::ObjectPtr SBMLModelObjectCache::store(const std::string &key, llvm::MemoryBufferRef obj) {
        ObjectPtr copy(llvm::MemoryBuffer::getMemBufferCopy(obj.getBuffer(), obj.getBufferIdentifier()));
        std::lock_guard<std::mutex> lock(mtx);
        cachedObjects[key] = copy;
        return copy;
    }

    std::vector<std::string> SBMLModelObjectCache::inspect() {
        std::vector<std::string> v;
        std::lock_guard<std::mutex> lock(mtx);
        rrLogDebug << "Number of cached models is: " << cachedObjects.size();
        for (auto &x: cachedObjects) {
            rrLogDebug << "Found cached model called: " << x.first().str();
            v.push_back(x.first().str());
        }
        return v;
    }

    void SBMLModelObjectCache::addToCache(const std::string &key, std::unique_ptr<llvm::MemoryBuffer> mb) {
        std::lock_guard<std::mutex> lock(mtx);
        cachedObjects.insert({key, ObjectPtr(std::move(mb))});
    }


}
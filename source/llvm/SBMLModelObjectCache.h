//
// Created by Ciaran on 05/11/2021.
//

#ifndef ROADRUNNER_SBMLMODELOBJECTCACHE_H
#define ROADRUNNER_SBMLMODELOBJECTCACHE_H

#define NOMINMAX

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif


namespace rrllvm {

    /**
     * @brief Stores SBML model as object files for later use.
     * @see llvm example called LLJitWithObjectCache.cpp
     * @details This object is a singleton. It cannot be copied, moved or assigned
     * and all interactions with this object must occur through the static call
     * SBMLModelObjectCache::getObjectCache();
     *
     * The key of an object is the identifier of the module it was compiled from. The
     * model generator sets it to the key of the model (see rr::getSBMLMD5), which holds
     * everything that changes the generated code. The code generation optimization
     * level of LLJit is one of these, because it is one of the load options.
     *
     * The objects are shared: an object that is replaced in the cache stays valid for
     * the Jit that uses it.
     *
     * All methods are thread safe.
     */
    class SBMLModelObjectCache : public llvm::ObjectCache {
    public:
        using ObjectPtr = std::shared_ptr<const llvm::MemoryBuffer>;

        static SBMLModelObjectCache& getObjectCache();

        SBMLModelObjectCache(const SBMLModelObjectCache& objectCache) = delete;

        SBMLModelObjectCache& operator=(const SBMLModelObjectCache&) = delete;

        SBMLModelObjectCache(SBMLModelObjectCache&& objectCache) noexcept = delete;

        SBMLModelObjectCache& operator=(SBMLModelObjectCache&&) noexcept= delete;

        void notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef ObjBuffer) override;

        std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *M) override;

        /**
         * @brief the object stored under @param key, or nullptr if there is none.
         */
        ObjectPtr find(const std::string& key);

        /**
         * @brief store a copy of @param obj under @param key and return it.
         * @details replaces the object that was stored under @param key before.
         */
        ObjectPtr store(const std::string& key, llvm::MemoryBufferRef obj);

        std::vector<std::string> inspect();

        void addToCache(const std::string & key, std::unique_ptr<llvm::MemoryBuffer> mb);

    private:
        SBMLModelObjectCache()= default;

        std::mutex mtx;

        llvm::StringMap<ObjectPtr> cachedObjects;

    };

    /**
     * @brief Instantiate a global instance of object cache.
     * @details This is experimental and may possible be better than
     * the one historically home rolled with roadrunner since
     * its actually an llvm construct. Notably, the compile layer
     * looks in the object cache prior to compiling. It also means that
     * we can access the object for load/save state. This is in contrast
     * to the old system whereby the IR is compiled to object file,
     * and manually stored as string before compiling the object file into
     * machine code.
     */
//    static ObjectCache objectCache;

}

#endif //ROADRUNNER_SBMLMODELOBJECTCACHE_H

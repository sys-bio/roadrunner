// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file gpu/Hasher.hpp
  * @author JKM
  * @date 09/07/2014
  * @copyright Apache License, Version 2.0
  * @brief Hash functions
**/

#ifndef rrGPUSimHasherH
#define rrGPUSimHasherH

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// #if (__cplusplus >= 201103L) || defined(_MSC_VER)
// #include <memory>
// #define cxx11_ns std
// #else
// #include <tr1/memory>
// #define cxx11_ns std::tr1
// #endif

// == INCLUDES ================================================

# include "gpu/sha1/TinySHA1.hpp"
# include <string>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

/**
 * @brief Hashed value
 */
class Hashval {
public:
    typedef sha1::SHA1::digest32_t digest32_t;

    /**
     * @brief Returns a string representation of the hash
     * @details No guarantees are made regarding the format of
     * the string or what base it is in
     */
    std::string str() const;

    /**
     * @brief Combines this hash value with another
     * @details Can be used to produce a hash for an
     * n-tuple. Works for the boundary case of repeated
     * values.
     */
    Hashval combined(const Hashval& other);

protected:
    Hashval() {}

    Hashval(sha1::SHA1& s);

    void useDigest(sha1::SHA1& s);

    friend class Hash;

    digest32_t digest_;
};

/**
 * @brief Hasher for basic types
 */
class Hash {
public:
    /// Return the hash of a std::string
    static Hashval me(const std::string& str);
};

} // namespace rrgpu

} // namespace rr
#endif /* rrGPUSimModelGeneratorH */

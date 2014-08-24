// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file gpu/bdom/BaseTypes.h
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief Base types used in code
  * @details void, int, etc.
**/

#ifndef rrGPU_BDOM_BaseTypes_H
#define rrGPU_BDOM_BaseTypes_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include "Serializer.hpp"

# include <memory>
# include <mutex>
# include <vector>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

/**
 * @author JKM
 * @brief Data type
 * @details Types are unique'd so type equality is deduced
 * via pointer comparison
 */
class Type {
public:
    typedef BDOM_String String;
    Type(const String& val, int tag)
      :  val_(val) {

    }

    virtual void serialize(Serializer& s) const {
        s << val_;
    }

protected:
    String val_;
};

/**
 * @author JKM
 * @brief Singleton for accessing base data types
 * @details There is only one instance of this class,
 * accessible via the @ref get method. The class keeps
 * track of all types
 */
class BaseTypes {
protected:
    typedef std::unique_ptr<BaseTypes> SelfPtr;
    typedef std::unique_ptr<Type> TypePtr;
    typedef std::vector<TypePtr> Types;
public:
    enum TYPE_TAG {
//         BASE_TYPES_BEGIN,
        VOID,
        INT,
        UNSIGNED_INT,
        SIZE_T,
        CSTR,
        BASE_TYPES_END
    };

    /** @brief Get the singleton instance
      */
    static BaseTypes& get() {
        std::call_once(once_, []{self_.reset(new BaseTypes());});
        return *self_;
    }

    /// Get the type associated with the given tag
    Type* getType(TYPE_TAG tag) {
        return types_.at(tag).get();
    }

    /// Fetch the singleton and get the type associated with the given tag
    static Type* getTp(TYPE_TAG tag) {
        return get().getType(tag);
    }

protected:
    static SelfPtr self_;
    static std::once_flag once_;
    Types types_;

private:
    // there can be only one!
    BaseTypes() {
        types_.emplace_back(new Type("void", VOID));
        types_.emplace_back(new Type("int", INT));
        types_.emplace_back(new Type("unsigned int", UNSIGNED_INT));
        types_.emplace_back(new Type("size_t", SIZE_T));
        types_.emplace_back(new Type("char*", SIZE_T));
    }

    BaseTypes(const BaseTypes& other);
    BaseTypes(BaseTypes&& other);
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

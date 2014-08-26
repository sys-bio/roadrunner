// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/BaseTypes.hpp
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
# include "patterns/Range.hpp"
# include "gpu/GPUSimException.h"

# include <memory>
# include <mutex>
# include <vector>
# include <list>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

template <class T>
using DomOwningPtr = std::unique_ptr<T>;

class TypeTransition;
class AliasType;

/**
 * @author JKM
 * @brief Data type
 * @details Types are unique'd (except for aliases) so type
 * equality is deduced via @ref isIdentical (which is a wrapper
 * for a dealiased pointer comparison)
 */
class Type {
protected:
    typedef std::list<TypeTransition*> Transitions;
    typedef Range<Transitions::iterator> TransitionRange;
    typedef Range<Transitions::const_iterator> TransitionConstRange;
    typedef std::list<AliasType*> Aliases;
    typedef Range<Aliases::iterator> AliasRange;
    typedef Range<Aliases::const_iterator> AliasConstRange;
public:
    typedef BDOM_String String;
    typedef std::size_t size_type;

    virtual ~Type() {}

    virtual void serialize(Serializer& s) const = 0;

    /// Before the variable name
    virtual void serializeFirstPart(Serializer& s) const { serialize(s); }

    /// After the variable name (e.g. [] for arrays)
    virtual void serializeSecondPart(Serializer& s) const {}

    TransitionRange transitions() { return TransitionRange(tx_); }

    TransitionConstRange transitions() const { return TransitionConstRange(tx_); }

    /// @internal
    void addTransition(TypeTransition* t) {
        tx_.emplace_back(t);
    }

    void addAlias(AliasType* other);

    AliasRange aliases() { return AliasRange(aliases_); }

    AliasConstRange aliases() const { return AliasConstRange(aliases_); }

    /// Used to remove aliases for type comparison
    virtual Type* dealias() { return this; }

    /// Used to remove aliases for type comparison
    virtual const Type* dealias() const { return this; }

    virtual bool isIdentical(const Type* other) const {
        return dealias() == other->dealias();
    }

protected:
    /// Used in @ref serialize to build representation (adding *'s and &'s)
//     virtual std::string buildRep();

    Transitions tx_;
    Type* root_ = nullptr;
    Aliases aliases_;
};

inline Serializer& operator<<(Serializer& s,  const Type& t) {
    t.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief Base data type
 * @details char, int, void, etc.
 */
class BaseType : public Type {
public:
    BaseType(const String& val, int tag)
      :  val_(val) {

    }

    virtual void serialize(Serializer& s) const;

protected:
    /// Used in @ref serialize to build representation (adding *'s and &'s)
//     virtual std::string buildRep();

    String val_;
};

/**
 * @author JKM
 * @brief A pointer to a type
 */
class PointerType : public Type {
public:
    PointerType(Type* root)
      : root_(root) {

    }

    Type* getRoot() {
        if (!root_)
            throw_gpusim_exception("Pointer has no root type");
        return root_;
    }

    const Type* getRoot() const {
        if (!root_)
            throw_gpusim_exception("Pointer has no root type");
        return root_;
    }

    virtual void serialize(Serializer& s) const;

protected:
    /// Non-owning
    Type* root_ = nullptr;
};

/**
 * @author JKM
 * @brief A pointer to a type
 */
class ArrayType : public Type {
public:
    ArrayType(Type* root)
      : root_(root) {

    }

    Type* getRoot() {
        if (!root_)
            throw_gpusim_exception("Array has no root type");
        return root_;
    }

    const Type* getRoot() const {
        if (!root_)
            throw_gpusim_exception("Array has no root type");
        return root_;
    }

    virtual void serialize(Serializer& s) const;

    /// Before the variable name
    virtual void serializeFirstPart(Serializer& s) const;

    /// After the variable name (e.g. [] for arrays)
    virtual void serializeSecondPart(Serializer& s) const;

protected:
    /// Non-owning
    Type* root_ = nullptr;
};

/**
 * @author JKM
 * @brief An alias to another type
 * @details Defined in e.g. a typedef
 */
class AliasType : public Type {
public:
    AliasType(Type* target, const std::string& alias)
      : target_(target), name_(alias) {

    }

    Type* getTarget() {
        if (!target_)
            throw_gpusim_exception("Alias has no target type");
        return target_;
    }

    const Type* getTarget() const {
        if (!target_)
            throw_gpusim_exception("Alias has no target type");
        return target_;
    }

    const std::string& getName() const { return name_; }

    virtual void serialize(Serializer& s) const;

    virtual Type* dealias() {
        return getTarget()->dealias();
    }

    virtual const Type* dealias() const {
        return getTarget()->dealias();
    }

protected:
    std::string name_;
    /// Non-owning
    Type* target_ = nullptr;
};
typedef DomOwningPtr<AliasType> AliasTypePtr;

/**
 * @author JKM
 * @brief Type transition
 * @details Add pointer etc.
 */
class TypeTransition {
public:
    TypeTransition(Type* from, Type* to)
      : from_(from), to_(to) {}

    virtual ~TypeTransition() {}

    TypeTransition(const TypeTransition&) = delete;
    TypeTransition(TypeTransition&&) = default;

    Type* getFrom() { return from_; }
    const Type* getFrom() const { return from_; }

    Type* getTo() { return to_; }
    const Type* getTo() const { return to_; }

    Type* getOpposite(const Type* t) {
        if(t == from_)
            return to_;
        else if(t == to_)
            return from_;
        else
            throw_gpusim_exception("No such participant type");
    }

    const Type* getOpposite(const Type* t) const {
        if(t == from_)
            return to_;
        else if(t == to_)
            return from_;
        else
            throw_gpusim_exception("No such participant type");
    }

    virtual bool isEquivalent(TypeTransition* other) = 0;

protected:
    /// Non-owning
    Type* from_ = nullptr;
    /// Non-owning
    Type* to_ = nullptr;
};
typedef DomOwningPtr<TypeTransition> TypeTransitionPtr;

/**
 * @author JKM
 * @brief Add pointer transition
 * @details Given a type, get the pointer type
 * (char -> char* etc.)
 */
class TypeTransitionAddPtr : public TypeTransition {
public:
    TypeTransitionAddPtr(Type* from, Type* to)
      : TypeTransition(from, to) {}

    virtual bool isEquivalent(TypeTransition* other) {
        if(auto x = dynamic_cast<TypeTransitionAddPtr*>(other))
            return getFrom()->isIdentical(x->getFrom());
        return false;
    }

protected:
};

/**
 * @author JKM
 * @brief Add array transition
 * @details Given a type, get the array type
 * (char -> char[] etc.)
 */
class TypeTransitionAddArray : public TypeTransition {
public:
    TypeTransitionAddArray(Type* from, Type* to)
      : TypeTransition(from, to) {}

    virtual bool isEquivalent(TypeTransition* other) {
        if(auto x = dynamic_cast<TypeTransitionAddArray*>(other))
            return getFrom()->isIdentical(x->getFrom());
        return false;
    }

protected:
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
    typedef DomOwningPtr<BaseTypes> SelfPtr;
    typedef DomOwningPtr<Type> TypePtr;
    typedef std::vector<TypePtr> Types;
    typedef std::vector<TypeTransitionPtr> Transitions;
public:
    enum TYPE_TAG {
//         BASE_TYPES_BEGIN,
        VOID,
        INT,
        UNSIGNED_INT,
        SIZE_T,
        FLOAT,
        DOUBLE,
        CHAR,
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

    /// Return the unique type formed by adding a pointer to an existing type
    Type* addPointer(Type* t) {
        TypePtr newtype(new PointerType(t));
        // if the type already exists, do not register a new one
        if(Type* preempt = regTransition(TypeTransitionPtr(new TypeTransitionAddPtr(t, newtype.get()))))
            return preempt;
        else
            return addType(std::move(newtype));
    }

    Type* newAliasType(Type* target, const std::string& alias) {
        AliasTypePtr a(new AliasType(target, alias));
        target->addAlias(a.get());
        return addType(std::move(a));
    }

    /// Return the unique type formed by making an array of an existing type
    Type* addArray(Type* t) {
        TypePtr newtype(new ArrayType(t));
        // if the type already exists, do not register a new one
        if(Type* preempt = regTransition(TypeTransitionPtr(new TypeTransitionAddArray(t, newtype.get()))))
            return preempt;
        else
            return addType(std::move(newtype));
    }

protected:

    /**
     * @brief Registers the transition if unique
     * @returns The preexisting target type if a transition already exists, nullptr otherwise
     */
    Type* regTransition(TypeTransitionPtr&& tx) {
        for(TypeTransition* t : tx->getFrom()->transitions())
            if(tx->isEquivalent(t))
                return t->getTo();
        tx->getFrom()->addTransition(tx.get());
        tx->getTo()->addTransition(tx.get());
        tx_.emplace_back(std::move(tx));
        return nullptr;
    }

    Type* addType(TypePtr&& t) {
        Type* result = t.get();
        types_.emplace_back(std::move(t));
        return result;
    }

    static SelfPtr self_;
    static std::once_flag once_;
    Types types_;
    Transitions tx_;

private:
    // there can be only one!
    BaseTypes() {
        types_.emplace_back(new BaseType("void", VOID));
        types_.emplace_back(new BaseType("int", INT));
        types_.emplace_back(new BaseType("unsigned int", UNSIGNED_INT));
        types_.emplace_back(new BaseType("size_t", SIZE_T));
        types_.emplace_back(new BaseType("float", FLOAT));
        types_.emplace_back(new BaseType("double", DOUBLE));
        types_.emplace_back(new BaseType("char", CHAR));
        addPointer(types_.back().get()); // CSTR
    }

    BaseTypes(const BaseTypes& other) = delete;
    BaseTypes(BaseTypes&& other) = delete;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

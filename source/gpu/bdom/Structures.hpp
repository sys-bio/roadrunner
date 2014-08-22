// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file gpu/bdom/Structures.h
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief Structural elements of code
  * @details functions, classes etc.
**/

#ifndef rrGPU_BDOM_Structures_H
#define rrGPU_BDOM_Structures_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include "BaseTypes.hpp"
# include "Expressions.hpp"
# include "patterns/AccessPtrIterator.hpp"
# include "patterns/Range.hpp"

# include <iostream>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

class Block {
public:
    Block() {}

    virtual void serialize(std::ostream& os) {}
};

class Function : public Block {
protected:
    typedef std::unique_ptr<Variable> VariablePtr;
    typedef std::vector<VariablePtr> Args;
public:
    typedef Type::String String;

    /// Ctor: initialize member vars
    Function(const String& name, Type* returnTp, std::initializer_list<Variable> args)
      : name_(name), returnTp_(returnTp) {
        for (const Variable& a : args)
            args_.emplace_back(new Variable(a));
    }

    virtual void serialize(std::ostream& os) const;

    typedef AccessPtrIterator<Args::iterator> ArgIterator;
    typedef AccessPtrIterator<Args::const_iterator> ConstArgIterator;

    typedef Range<ArgIterator> ArgRange;
    typedef Range<ConstArgIterator> ConstArgRange;

    ArgRange getArgs() { return ArgRange(args_); }
    ConstArgRange getArgs() const { return ConstArgRange(args_); }

protected:
    String name_;
    Type* returnTp_;
    Args args_;
};

/**
 * @author JKM
 * @brief module
 */
class Module {
protected:
    typedef std::unique_ptr<Function> FunctionPtr;
    typedef std::vector<FunctionPtr> Functions;
    Functions func_;
public:
    /// Serialize to a source file
    virtual void serialize(std::ostream& os) = 0;
};


} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

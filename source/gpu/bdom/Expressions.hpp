// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file gpu/bdom/Expressions.h
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief Expressions
  * @details if (x != 1) { ++y; }
**/

#ifndef rrGPU_BDOM_Expressions_H
#define rrGPU_BDOM_Expressions_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include "BaseTypes.hpp"

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

class Variable {
public:
    typedef Type::String String;
    Variable(const String& name, Type* type)
      : name_(name), type_(type) {

    }

    virtual void serialize(std::ostream& os) const {
        type_->serialize(os);
        os << String(" ");
        os << name_;
    }

protected:
    String name_;
    Type* type_ = nullptr;
};

class Expression {
public:
    Expression() {}
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

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
# include "gpu/GPUSimException.h"

# include <unordered_map>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

/**
 * @author JKM
 * @brief A class to encapsulate variable declarations
 */
class Variable {
public:
    ~Variable() {}

    typedef Type::String String;
    Variable(const String& name, Type* type)
      : name_(name), type_(type) {

    }

    virtual void serialize(Serializer& s) const {
        type_->serialize(s);
        s << String(" ");
        s << name_;
    }

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

protected:
    String name_;
    Type* type_ = nullptr;
};

/**
 * @author JKM
 * @brief A function parameter is a subclass of variable
 */
class FunctionParameter : public Variable {
public:
    using Variable::Variable;

protected:
};

/**
 * @author JKM
 * @brief A class to encapsulate expressions
 */
class Expression {
public:
    Expression() {}
    ~Expression() {}

    virtual void serialize(Serializer& s) const = 0;
};
typedef std::unique_ptr<Expression> ExpressionPtr;

class Function;

/**
 * @author JKM
 * @brief arg map
 */
// class FunctionArgMap {
// public:
//     /// Does not take ownership
//     FunctionCallExpression(Function*) {}
//
// protected:
//     std::unordered_map<Variable*>
// };

/**
 * @author JKM
 * @brief A class to encapsulate expressions
 */
class FunctionCallExpression : public Expression {
public:
    /// Does not take ownership
    FunctionCallExpression(Function* func)
      :  func_(func) {}

    /// Pass the var @ref v as the arg @ref p of the function
    void passArgument(FunctionParameter* p, Variable* v);

    virtual void serialize(Serializer& s) const;

protected:
    /// non-owning
    Function* func_;
    typedef std::unordered_map<FunctionParameter*, Variable*> FunctionArgMap;
    FunctionArgMap argmap_;
};

/**
 * @author JKM
 * @brief A statement
 */
class Statement {
public:
    Statement() {}
    ~Statement() {}

    virtual void serialize(Serializer& s) const = 0;
};
typedef std::unique_ptr<Statement> StatementPtr;

/**
 * @author JKM
 * @brief A statement containing an expression
 */
class ExpressionStatement : public Statement {
public:

    ExpressionStatement(ExpressionPtr&& exp)
      : exp_(std::move(exp)) {}

    Expression* getExpression() {
        if (!exp_)
            throw_gpusim_exception("No expression");
        return exp_.get();
    }

    const Expression* getExpression() const {
        if (!exp_)
            throw_gpusim_exception("No expression");
        return exp_.get();
    }

    virtual void serialize(Serializer& s) const;

protected:
    ExpressionPtr exp_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

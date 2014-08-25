// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/Expressions.hpp
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
typedef std::unique_ptr<FunctionParameter> FunctionParameterPtr;

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

inline Serializer& operator<<(Serializer& s,  const Expression& e) {
    e.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief A class to encapsulate literals
 */
class LiteralExpression : public Expression {
public:
    LiteralExpression() {}

    virtual void serialize(Serializer& s) const = 0;
};

/**
 * @author JKM
 * @brief A class to encapsulate literals
 */
//TODO: rename IntLiteralExpression
class LiteralIntExpression : public Expression {
public:
    LiteralIntExpression(int i) : i_(i) {}

    virtual void serialize(Serializer& s) const;

protected:
    int i_=0;
};

/**
 * @author JKM
 * @brief A class to encapsulate string literals
 */
class StringLiteralExpression : public Expression {
public:
    StringLiteralExpression(const std::string s) : s_(s) {}

    virtual void serialize(Serializer& s) const;

protected:
    std::string s_;
};

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
 * @brief A function call expression
 * @details Represents a call to a function. Expressions
 * may be used to pass parameters.
 */
class FunctionCallExpression : public Expression {
public:
    /// Does not take ownership
    FunctionCallExpression(const Function* func)
      :  func_(func) {}

    /// Takes ownership of args (direct all outcry to the C++ committee)
//     FunctionCallExpression(const Function* func, std::initializer_list<Expression*> args);

    FunctionCallExpression(const Function* func, ExpressionPtr&& u)
      :  func_(func) {
        passMappedArgument(std::move(u), getPositionalParam(0));
    }

    /// Pass the exp @ref v as the arg @ref p of the function
    void passMappedArgument(ExpressionPtr&& v, const FunctionParameter* p);

    const FunctionParameter* getPositionalParam(int i) const;

    /// Serialize this object to a document
    virtual void serialize(Serializer& s) const;

protected:
    /// Target function (non-owning)
    const Function* func_;
    typedef std::unordered_map<const FunctionParameter*, ExpressionPtr> FunctionArgMap;
    /// Arguments
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

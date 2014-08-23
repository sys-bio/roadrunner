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
protected:
    typedef std::vector<StatementPtr> Statements;
public:
    Block() {}
    Block(const Block&) = delete;
    Block(Block&&) = default;
    ~Block();

    void addStatement(StatementPtr&& stmt) {
        stmts_.emplace_back(std::move(stmt));
    }

    typedef AccessPtrIterator<Statements::iterator> StatementIterator;
    typedef AccessPtrIterator<Statements::const_iterator> ConstStatementIterator;

    typedef Range<StatementIterator> StatementRange;
    typedef Range<ConstStatementIterator> ConstStatementRange;

    StatementRange getStatements() { return StatementRange(stmts_); }
    ConstStatementRange getStatements() const { return ConstStatementRange(stmts_); }

    virtual void serialize(Serializer& s) const;

protected:
    Statements stmts_;
};

class Function : public Block {
protected:
    typedef std::unique_ptr<FunctionParameter> FunctionParameterPtr;
    typedef std::vector<FunctionParameterPtr> Args;
public:
    typedef Type::String String;

    /// Ctor: initialize member vars
    Function(const String& name, Type* returnTp)
      : name_(name), returnTp_(returnTp) {
    }

    /// Ctor: initialize member vars with arg list
    Function(const String& name, Type* returnTp, std::initializer_list<FunctionParameter> args)
      : Function(name, returnTp) {
        for (const FunctionParameter& a : args)
            args_.emplace_back(new FunctionParameter(a));
    }

    virtual void serialize(Serializer& s) const;

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

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
    virtual void serialize(Serializer& s) const = 0;

    typedef AccessPtrIterator<Functions::iterator> FunctionIterator;
    typedef AccessPtrIterator<Functions::const_iterator> ConstFunctionIterator;

    typedef Range<FunctionIterator> FunctionRange;
    typedef Range<ConstFunctionIterator> ConstFunctionRange;

    FunctionRange getFunctions() { return FunctionRange(func_); }
    ConstFunctionRange getFunctions() const { return ConstFunctionRange(func_); }
};


} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

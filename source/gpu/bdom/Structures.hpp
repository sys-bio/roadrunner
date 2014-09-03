// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/Structures.hpp
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

/**
 * @author JKM
 * @brief A block of code
 * @details Blocks are the basic unit of code.
 * They appear in function bodies, if statements,
 * loops, etc.
 */
class Block : public StatementContainer {
public:
    Block() {}
    Block(const Block&) = delete;
    Block(Block&&) = default;
    ~Block() {}

    virtual void serialize(Serializer& s) const;
};

inline Serializer& operator<<(Serializer& s,  const Block& b) {
    b.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief A statement containing a for loop
 */
class ForStatement : public Statement, public Scope {
protected:
    typedef std::vector<VariablePtr> Variables;
public:
    /// Empty ctor
    ForStatement();

    /// Ctor for typedef @a target @a alias
    ForStatement(ExpressionPtr&& init_exp, ExpressionPtr&& cond_exp, ExpressionPtr&& loop_exp);

    /// Shortcut for constructing an owning pointer
    static DomOwningPtr<ForStatement> make() {
        return DomOwningPtr<ForStatement>(new ForStatement());
    }

    /// Shortcut for inserting into a code block / module
    static ForStatement* insert(StatementContainer& c) {
        return downcast(c.addStatement(make()));
    }

    /// Shortcut for inserting into a code block / module
    static ForStatement* insert(StatementContainer* c) {
        return downcast(c->addStatement(make()));
    }

    Expression* getInitExp() { return init_exp_.get(); }
    const Expression* getInitExp() const { return init_exp_.get(); }
    void setInitExp(ExpressionPtr&& init_exp) { init_exp_ = std::move(init_exp); }

    Expression* getCondExp() { return cond_exp_.get(); }
    const Expression* getCondExp() const { return cond_exp_.get(); }
    void setCondExp(ExpressionPtr&& cond_exp) { cond_exp_ = std::move(cond_exp); }

    Expression* getLoopExp() { return loop_exp_.get(); }
    const Expression* getLoopExp() const { return loop_exp_.get(); }
    void setLoopExp(ExpressionPtr&& loop_exp) { loop_exp_ = std::move(loop_exp); }

    Block& getBody() { return body_; }
    const Block& getBody() const { return body_; }

    Variable* addVariable(VariablePtr&& var) {
        vars_.emplace_back(std::move(var));
        return vars_.back().get();
    }

    Variable* addVariable(Variable&& var) {
        return addVariable(VariablePtr(new Variable(std::move(var))));
    }

    virtual void serialize(Serializer& s) const;

    // TODO: replace with LLVM-style casting
    static ForStatement* downcast(Statement* s) {
        auto result = dynamic_cast<ForStatement*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    ExpressionPtr init_exp_;
    ExpressionPtr cond_exp_;
    ExpressionPtr loop_exp_;

    Block body_;
    Variables vars_;
};
typedef DomOwningPtr<ForStatement> ForStatementPtr;

/**
 * @author JKM
 * @brief A statement containing a typedef
 */
class IfStatement : public Statement {
public:
    /// Ctor for typedef @a target @a alias
    IfStatement(ExpressionPtr&& condition);

    Expression* getCondExp() { return condition_.get(); }
    const Expression* getCondExp() const { return condition_.get(); }
    void setCondExp(ExpressionPtr&& cond_exp) { condition_ = std::move(cond_exp); }

    Block& getBody() { return body_; }
    const Block& getBody() const { return body_; }

    virtual void serialize(Serializer& s) const;

    // TODO: replace with LLVM-style casting
    static IfStatement* downcast(Statement* s) {
        auto result = dynamic_cast<IfStatement*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    ExpressionPtr condition_;

    Block body_;
};
typedef DomOwningPtr<IfStatement> IfStatementPtr;

/**
 * @author JKM
 * @brief A label
 */
class Label : public Statement {
public:

    virtual void serialize(Serializer& s) const = 0;

    // TODO: replace with LLVM-style casting
    static Label* downcast(Statement* s) {
        auto result = dynamic_cast<Label*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

};
typedef DomOwningPtr<Label> LabelPtr;

/**
 * @author JKM
 * @brief A case label
 */
class CaseLabel : public Label {
public:
    CaseLabel(ExpressionPtr&& val)
      : val_(std::move(val)) {}

    /// Copy ctor
    CaseLabel(const CaseLabel& other)
      : val_(other.val_->clone()) {}

    const Expression* getValue() const { return val_.get(); }

    virtual void serialize(Serializer& s) const;

    virtual DomOwningPtr<CaseLabel> clone() const {
        return DomOwningPtr<CaseLabel>(new CaseLabel(*this));
    }

protected:
    ExpressionPtr val_;
};
typedef DomOwningPtr<CaseLabel> CaseLabelPtr;

/**
 * @author JKM
 * @brief A case label
 */
class DefaultLabel : public Label {
public:
    DefaultLabel() {}

    /// Copy ctor
    DefaultLabel(const DefaultLabel& other) {}

    virtual void serialize(Serializer& s) const;

    virtual DomOwningPtr<DefaultLabel> clone() const {
        return DomOwningPtr<DefaultLabel>(new DefaultLabel(*this));
    }
};
typedef DomOwningPtr<DefaultLabel> DefaultLabelPtr;

/**
 * @author JKM
 * @brief A statement containing a switch
 */
class SwitchStatement : public Statement, public Scope {
public:
    /// Empty ctor
    SwitchStatement(ExpressionPtr&& exp);

    /// Shortcut for constructing an owning pointer
    static DomOwningPtr<SwitchStatement> make(ExpressionPtr&& exp) {
        return DomOwningPtr<SwitchStatement>(new SwitchStatement(std::move(exp)));
    }

    /// Shortcut for inserting into a code block / module
    static SwitchStatement* insert(StatementContainer& c, ExpressionPtr&& exp) {
        return downcast(c.addStatement(make(std::move(exp))));
    }

    /// Shortcut for inserting into a code block / module
    template <class SwitchExpressionT>
    static SwitchStatement* insert(StatementContainer& c, SwitchExpressionT&& exp) {
        return insert(c, ExpressionPtr(new SwitchExpressionT(exp)));
    }

    /// Shortcut for inserting into a code block / module
    static SwitchStatement* insert(StatementContainer* c, ExpressionPtr&& exp) {
        return downcast(c->addStatement(make(std::move(exp))));
    }
    Expression* getExpression() const {
        if (!exp_)
            throw_gpusim_exception("No switch variable set");
        return exp_.get();
    }

    Block& getBody() { return body_; }
    const Block& getBody() const { return body_; }

    Label* addCase(ExpressionPtr&& val) {
        return Label::downcast(getBody().addStatement(StatementPtr(new CaseLabel(std::move(val)))));
    }

    template <class ValueExpressionT>
    Label* addCase(ValueExpressionT&& val) {
        return addCase(ExpressionPtr(new ValueExpressionT(std::move(val))));
    }

    Label* addDefault() {
        cases_.emplace_back(new DefaultLabel());
        return cases_.back().get();
    }

    void addBreak() {
        getBody().addStatement(StatementPtr(new BreakStatement()));
    }

    virtual void serialize(Serializer& s) const;

    // TODO: replace with LLVM-style casting
    static SwitchStatement* downcast(Statement* s) {
        auto result = dynamic_cast<SwitchStatement*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    ExpressionPtr exp_ = nullptr;
    Block body_;
    typedef std::vector<LabelPtr> Cases;
    Cases cases_;
};
typedef DomOwningPtr<SwitchStatement> SwitchStatementPtr;

/**
 * @author JKM
 * @brief A function
 */
class Function : public Block {
protected:
//     typedef DomOwningPtr<FunctionParameter> FunctionParameterPtr;
    typedef std::vector<FunctionParameterPtr> Args;
public:
    typedef Type::String String;
    typedef FunctionCallExpression::size_type size_type;

    /// Ctor: name / type
    Function(const String& name, Type* returnTp)
      : name_(name), returnTp_(returnTp) {
    }

    /// One-arg ctor
    Function(const String& name, Type* returnTp, FunctionParameterPtr&& u)
      : name_(name), returnTp_(returnTp) {
        args_.emplace_back(std::move(u));
    }

    /// Ctor: initialize member vars with arg list
    Function(const String& name, Type* returnTp, std::initializer_list<FunctionParameter> args)
      : Function(name, returnTp) {
        for (const FunctionParameter& a : args)
            args_.emplace_back(new FunctionParameter(a));
    }

    virtual bool requiresSpecialCallingConvention() const { return false; }

    /// Get the name of the function
    const String& getName() const { return name_; }
    /// Set the name of the function
    void setName(const String& name) { name_ = name; }

    /// Declared with 'extern "C"'?
    bool hasCLinkage() const { return clink_; }
    /// Set whether func has C linkage (will be serialized with 'extern "C"')
    void setHasCLinkage(bool val) { clink_ = val; }

    /// Get the number of regular positional parameters
    size_type getNumPositionalParams() const {
        return args_.size();
    }

    //TODO: rename arg -> param
    typedef AccessPtrIterator<Args::iterator> ArgIterator;
    typedef AccessPtrIterator<Args::const_iterator> ConstArgIterator;

    typedef Range<ArgIterator> ArgRange;
    typedef Range<ConstArgIterator> ConstArgRange;

    ArgRange getArgs() { return ArgRange(args_); }
    ConstArgRange getArgs() const { return ConstArgRange(args_); }

    /// Get the function parameter at position @ref i (zero-offset)
    const FunctionParameter* getPositionalParam(int i) const {
        return args_.at(i).get();
    }

    /// Return true if variadic function
    bool isVarargs() const {
        return is_varargs_;
    }

    void setIsVarargs(bool val) {
        is_varargs_ = val;
    }

    /// Serialize method
    virtual void serialize(Serializer& s) const;

protected:
    /// Serialize the header only
    void serializeHeader(Serializer& s) const;

    String name_;
    Type* returnTp_;
    Args args_;
    bool clink_ = false;
    bool is_varargs_ = false;
};
typedef DomOwningPtr<Function> FunctionPtr;

/**
 * @author JKM
 * @brief module
 */
// TODO: derive both this and Block from a common ancestor
class Module : public StatementContainer {
protected:
    typedef DomOwningPtr<Function> FunctionPtr;
    typedef std::vector<FunctionPtr> Functions;
    Functions func_;
    typedef std::vector<MacroPtr> Macros;
    Macros macros_;
public:
    /// Serialize to a source file
    virtual void serialize(Serializer& s) const = 0;

    typedef AccessPtrIterator<Functions::iterator> FunctionIterator;
    typedef AccessPtrIterator<Functions::const_iterator> ConstFunctionIterator;

    typedef Range<FunctionIterator> FunctionRange;
    typedef Range<ConstFunctionIterator> ConstFunctionRange;

    FunctionRange getFunctions() { return FunctionRange(func_); }
    ConstFunctionRange getFunctions() const { return ConstFunctionRange(func_); }

    Macro* addMacro(MacroPtr&& mac) {
        macros_.emplace_back(std::move(mac));
        return macros_.back().get();
    }

    Macro* addMacro(Macro&& mac) {
        macros_.emplace_back(new Macro(std::move(mac)));
        return macros_.back().get();
    }

    typedef AccessPtrIterator<Macros::iterator> MacroIterator;
    typedef AccessPtrIterator<Macros::const_iterator> ConstMacroIterator;

    typedef Range<MacroIterator> MacroRange;
    typedef Range<ConstMacroIterator> ConstMacroRange;

    MacroRange getMacros() { return MacroRange(macros_); }
    ConstMacroRange getMacros() const { return ConstMacroRange(macros_); }

    Function* getSizeof() {
        return sizeof_.get();
    }

    const Function* getSizeof() const {
        return sizeof_.get();
    }

protected:
    void serializeMacros(Serializer& s) const;

    void serializeStatements(Serializer& s) const {
        StatementContainer::serialize(s);
    }

    FunctionPtr sizeof_{new Function("sizeof",
                            BaseTypes::getTp(BaseTypes::SIZE_T),
                            FunctionParameterPtr(new FunctionParameter(BaseTypes::getTp(BaseTypes::ANY), "exp")))};
};


} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

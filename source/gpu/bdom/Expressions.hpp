// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/Expressions.hpp
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief Expressions
  * @details Example: if (x != 1) { ++y; }
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

// hack to get needed functions (redo with CRTP later maybe)
# define INSERT_STD_MAKE_CODE(TypeName)                \
                                                       \
    /// Shortcut for constructing an owning pointer    \
    static DomOwningPtr<TypeName> make() {             \
        return DomOwningPtr<TypeName>(new TypeName()); \
    }

/// Use insert(container, ctor_args...) to insert a statement into a container (e.g. @ref Block)
# define INSERT_STD_STMT_CODE(TypeName)                                                            \
    /* Shortcut for inserting into a code block / module */                                        \
    template<typename... Args>                                                                     \
    static TypeName* insert(StatementContainer& c, Args&&... args) {                               \
        return downcast(c.addStatement(StatementPtr(new TypeName(std::forward<Args>(args)...))));  \
    }                                                                                              \
                                                                                                   \
    /* Shortcut for inserting into a code block / module */                                        \
    template<typename... Args>                                                                     \
    static TypeName* insert(StatementContainer* c, Args&&... args) {                               \
        return downcast(c->addStatement(StatementPtr(new TypeName(std::forward<Args>(args)...)))); \
    }                                                                                              \
                                                                                                   \
    /* TODO: replace with LLVM-style casting */                                                    \
    static TypeName* downcast(Statement* s) {                                                      \
        auto result = dynamic_cast<TypeName*>(s);                                                  \
        if (!result)                                                                               \
            throw_gpusim_exception("Downcast failed: incorrect type");                             \
        return result;                                                                             \
    }

// == INCLUDES ================================================

# include "BaseTypes.hpp"
# include "Preproc.hpp"
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
 * @brief A scope in which variables etc. are defined
 */
class Scope {
public:

};

/**
 * @author JKM
 * @brief A class to encapsulate variables
 */
class Variable {
public:
    typedef Type::String String;
    Variable(Type* type, const String& name)
      : name_(name), type_(type) {

    }

    virtual ~Variable() {}

    virtual void serialize(Serializer& s) const;

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

    Type* getType() {
        if (!type_)
            throw_gpusim_exception("No type set for variable");
        return type_;
    }
    const Type* getType() const {
        if (!type_)
            throw_gpusim_exception("No type set for variable");
        return type_;
    }

    Scope* getScope() {
        if (!scope_)
            throw_gpusim_exception("No scope set for variable");
        return scope_;
    }
    const Scope* getScope() const {
        if (!scope_)
            throw_gpusim_exception("No scope set for variable");
        return scope_;
    }
    void setScope(Scope* scope) {
        scope_ = scope;
    }

protected:
    String name_;
    Type* type_ = nullptr;
    Scope* scope_ = nullptr;
};
typedef DomOwningPtr<Variable> VariablePtr;

inline Serializer& operator<<(Serializer& s, const Variable& v) {
    v.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief A function parameter is a subclass of variable
 */
class FunctionParameter : public Variable {
public:
    using Variable::Variable;

protected:
};
typedef DomOwningPtr<FunctionParameter> FunctionParameterPtr;

/**
 * @author JKM
 * @brief A class to encapsulate expressions
 */
class Expression {
public:
    typedef Type::String String;

    Expression() {}
    virtual ~Expression() {}

    virtual void serialize(Serializer& s) const = 0;
};
typedef DomOwningPtr<Expression> ExpressionPtr;

inline Serializer& operator<<(Serializer& s,  const Expression& e) {
    e.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief A class to encapsulate expressions
 */
class EmptyExpression : public Expression {
public:
    EmptyExpression() {}

    virtual void serialize(Serializer& s) const {}
};

/**
 * @author JKM
 * @brief An expression to simply reference a variable
 * @details Just serializes the variable's name
 */
class VariableRefExpression : public Expression {
public:
    /// Ctor for type, var name, and initial value
    VariableRefExpression(Variable* var)
      : var_(var) {}

    Variable* getVariable() {
        if (!var_)
            throw_gpusim_exception("No variable set");
        return var_;
    }
    const Variable* getVariable() const {
        if (!var_)
            throw_gpusim_exception("No variable set");
        return var_;
    }

    virtual void serialize(Serializer& s) const;

protected:
    Variable* var_;
};

/**
 * @author JKM
 * @brief An expression to simply reference a variable
 * @details Just serializes the variable's name
 */
class ArrayIndexExpression : public VariableRefExpression {
public:
    /// Ctor for type, var name, and initial value
    ArrayIndexExpression(Variable* var, ExpressionPtr&& index_exp)
      : VariableRefExpression(var), index_exp_(std::move(index_exp)) {}

    template <class ExpressionT>
    ArrayIndexExpression(Variable* var, ExpressionT index_exp)
      : ArrayIndexExpression(var, ExpressionPtr(new ExpressionT(std::move(index_exp)))) {}

    /// Get the expression used to index into the array
    Expression* getIndexExp() {
        if (!index_exp_)
            throw_gpusim_exception("No index expression set");
        return index_exp_.get();
    }
    /// Get Expression expression used to index into the array
    const Expression* getIndexExp() const {
        if (!index_exp_)
            throw_gpusim_exception("No index expression set");
        return index_exp_.get();
    }

    virtual void serialize(Serializer& s) const;

protected:
    Variable* var_;
    ExpressionPtr index_exp_;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class VariableDeclarationExpression : public Expression {
public:
    /// Ctor
    VariableDeclarationExpression(Variable* var)
      : var_(var) {}

    Type* getType() { return getVariable()->getType(); }
    const Type* getType() const { return getVariable()->getType(); }

    Variable* getVariable() {
        if (!var_)
            throw_gpusim_exception("No variable set");
        return var_;
    }
    const Variable* getVariable() const {
        if (!var_)
            throw_gpusim_exception("No variable set");
        return var_;
    }

    virtual void serialize(Serializer& s) const;

protected:
    Variable* var_;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class VariableInitExpression : public VariableDeclarationExpression {
public:
    /// Ctor for type, var name, and initial value
    VariableInitExpression(Variable* var, ExpressionPtr&& value_exp)
      : VariableDeclarationExpression(var) {
        value_ = std::move(value_exp);
    }

    Expression* getValue() {
        if (!value_)
            throw_gpusim_exception("No variable set");
        return value_.get();
    }
    const Expression* getValue() const {
        if (!value_)
            throw_gpusim_exception("No variable set");
        return value_.get();
    }

    virtual void serialize(Serializer& s) const;

protected:
    ExpressionPtr value_;
};

/**
 * @author JKM
 * @brief References a macro
 */
class MacroExpression : public Expression {
public:
    /// Ctor
    MacroExpression(Macro* mac)
      : mac_(mac) {}

    Macro* getMacro() {
        if (!mac_)
            throw_gpusim_exception("No macro set");
        return mac_;
    }
    const Macro* getMacro() const {
        if (!mac_)
            throw_gpusim_exception("No macro set");
        return mac_;
    }

    virtual void serialize(Serializer& s) const;

protected:
    Macro* mac_;
};

/**
 * @author JKM
 * @brief Unary operators
 */
class UnaryExpression : public Expression {
public:
    UnaryExpression(ExpressionPtr&& operand)
      : operand_(std::move(operand)) {
    }

    Expression* getOperand() {
        if (!operand_)
            throw_gpusim_exception("No LHS set");
        return operand_.get();
    }
    const Expression* getOperand() const {
        if (!operand_)
            throw_gpusim_exception("No LHS set");
        return operand_.get();
    }

    virtual void serialize(Serializer& s) const = 0;

protected:
    ExpressionPtr operand_;
};

/**
 * @author JKM
 * @brief Pre-increment operator
 * @details e.g. ++j
 */
class PreincrementExpression : public UnaryExpression {
public:
    PreincrementExpression(ExpressionPtr&& operand)
      : UnaryExpression(std::move(operand)) {
    }

    virtual void serialize(Serializer& s) const;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class BinaryExpression : public Expression {
public:
    BinaryExpression(ExpressionPtr&& lhs, ExpressionPtr&& rhs)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    }

    template <class LHSExpression, class RHSExpression>
    BinaryExpression(LHSExpression&& lhs, RHSExpression&& rhs)
      : lhs_(new LHSExpression(std::move(lhs))), rhs_(new RHSExpression(std::move(rhs))) {
    }

    Expression* getLHS() {
        if (!lhs_)
            throw_gpusim_exception("No LHS set");
        return lhs_.get();
    }
    const Expression* getLHS() const {
        if (!lhs_)
            throw_gpusim_exception("No LHS set");
        return lhs_.get();
    }

    Expression* getRHS() {
        if (!rhs_)
            throw_gpusim_exception("No RHS set");
        return rhs_.get();
    }
    const Expression* getRHS() const {
        if (!rhs_)
            throw_gpusim_exception("No RHS set");
        return rhs_.get();
    }

    virtual void serialize(Serializer& s) const = 0;

protected:
    ExpressionPtr lhs_;
    ExpressionPtr rhs_;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class AssignmentExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;

    virtual void serialize(Serializer& s) const;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class LTComparisonExpression : public BinaryExpression {
public:
    /// Ctor for lhs & rhs
    LTComparisonExpression(ExpressionPtr&& lhs, ExpressionPtr&& rhs)
      : BinaryExpression(std::move(lhs), std::move(rhs)) {
    }

    virtual void serialize(Serializer& s) const;
};

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
    typedef std::size_t size_type;

    /// Does not take ownership
    FunctionCallExpression(const Function* func)
      :  func_(func) {}

    /// Takes ownership of args (direct all outcry to the C++ committee)
//     FunctionCallExpression(const Function* func, std::initializer_list<Expression*> args);

    /// One-arg ctor
    FunctionCallExpression(const Function* func, ExpressionPtr&& u)
      :  func_(func) {
        passMappedArgument(std::move(u), getPositionalParam(0));
    }

    /// N-arg ctor
    template <class... Args>
    FunctionCallExpression(const Function* func, Args&&... args)
      :  func_(func) {
        passArguments(std::forward<Args>(args)...);
    }

    /// Pass the exp @ref v as the arg @ref p of the function
    void passMappedArgument(ExpressionPtr&& v, const FunctionParameter* p);

    bool hasMappedArgument(const FunctionParameter* p) const {
        return argmap_.count(p);
    }

    const FunctionParameter* getPositionalParam(int i) const;

    size_type getNumPositionalParams() const;

    bool isVarargs() const;

    /// Pass @ref v as the next positional or variadic argument
    void passArgument(ExpressionPtr&& v);

    /**
     * @brief Pass @ref v as the next positional or variadic argument
     * @details Automatically creates an owning pointer and passes it
     * to the other signature
     */
    template <class ExpressionT>
    void passArgument(ExpressionT&& v) {
        passArgument(ExpressionPtr(new ExpressionT(std::move(v))));
    }

    /**
     * @brief Pass @ref v as the next positional or variadic argument
     * @details N-ary version for use with corresponding ctor
     */
    template <class ExpressionT, class... Args>
    void passArguments(ExpressionT&& arg, Args&&... args) {
        passArgument(std::move(arg));
        passArguments(std::forward<Args>(args)...);
    }

    void passArguments() {}

    /// Serialize this object to a document
    virtual void serialize(Serializer& s) const;

protected:
    void passExtraArg(ExpressionPtr&& v) {
        extra_args_.emplace_back(std::move(v));
    }

    /// Target function (non-owning)
    const Function* func_;
    typedef std::unordered_map<const FunctionParameter*, ExpressionPtr> FunctionArgMap;
    /// Arguments
    FunctionArgMap argmap_;

    typedef std::vector<ExpressionPtr> ExtraArgs;
    /// Extra arguments for variadic functions
    ExtraArgs extra_args_;
};

class Statement;
typedef DomOwningPtr<Statement> StatementPtr;

/**
 * @author JKM
 * @brief Superclass of block / module
 */
class StatementContainer {
protected:
    typedef std::vector<StatementPtr> Statements;
    typedef std::vector<VariablePtr> Variables;
public:
    StatementContainer() {}
    StatementContainer(const StatementContainer&) = delete;
    StatementContainer(StatementContainer&&) = default;
    virtual ~StatementContainer() {}

    Statement* addStatement(StatementPtr&& stmt);

    /// Convert @ref exp to a statement
    Statement* addStatement(ExpressionPtr&& exp);

    typedef AccessPtrIterator<Statements::iterator> StatementIterator;
    typedef AccessPtrIterator<Statements::const_iterator> ConstStatementIterator;

    typedef Range<StatementIterator> StatementRange;
    typedef Range<ConstStatementIterator> ConstStatementRange;

    StatementRange getStatements() { return StatementRange(stmts_); }
    ConstStatementRange getStatements() const { return ConstStatementRange(stmts_); }

    Variable* addVariable(VariablePtr&& var) {
        vars_.emplace_back(std::move(var));
        return vars_.back().get();
    }

    Variable* addVariable(Variable&& var) {
        return addVariable(VariablePtr(new Variable(std::move(var))));
    }

    virtual void serialize(Serializer& s) const;

protected:
    Statements stmts_;
    Variables vars_;
};

/**
 * @author JKM
 * @brief A statement
 */
class Statement {
public:
    Statement() {}
    virtual ~Statement() {}

    virtual void serialize(Serializer& s) const = 0;
};

/**
 * @author JKM
 * @brief A statement containing an expression
 */
class ExpressionStatement : public Statement {
public:

    ExpressionStatement(ExpressionPtr&& exp)
      : exp_(std::move(exp)) {}

    template <class ExpressionT>
    ExpressionStatement(ExpressionT&& exp)
      : exp_(new ExpressionT(std::move(exp))) {}

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

    INSERT_STD_STMT_CODE(ExpressionStatement)

    virtual void serialize(Serializer& s) const;

protected:
    ExpressionPtr exp_;
};

/**
 * @author JKM
 * @brief A statement containing a typedef
 */
class TypedefStatement : public Statement {
public:
    /// Ctor for typedef @a target @a alias
    TypedefStatement(Type* target, const std::string& alias);

    Type* getTarget() { return target_; }
    const Type* getTarget() const { return target_; }

    Type* getAlias() { return alias_; }
    const Type* getAlias() const { return alias_; }

    virtual void serialize(Serializer& s) const;

    static TypedefStatement* downcast(Statement* s) {
        TypedefStatement* result = dynamic_cast<TypedefStatement*>(s);
        if (!result)
            throw_gpusim_exception("Not a TypedefStatement");
        return result;
    }

protected:
    Type* target_;
    Type* alias_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

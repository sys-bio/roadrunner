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

    /// Deep copy
    virtual DomOwningPtr<Expression> clone() const = 0;
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

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new EmptyExpression());
    }
};

/**
 * @author JKM
 * @brief An expression to simply reference a variable
 * @details Just serializes the variable's name
 */
class SymbolExpression : public Expression {
public:
    SymbolExpression() {}
    /**
     * @brief Ctor
     * @param[in] sym The symbol string
     */
    SymbolExpression(const String& sym)
      : sym_(sym) {}

    /// Copy ctor
    SymbolExpression(const SymbolExpression& other)
      : sym_(other.sym_) {}

    /**
     * @brief Returns the symbol string
     * @note May be computed differently in derived classes
     * (e.g. @ref VariableRefExpression uses the variable name
     * instead of @ref sym_)
     */
    virtual String getSymbol() const {
        if (!sym_.size())
            throw_gpusim_exception("No symbol set");
        return sym_;
    }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new SymbolExpression(*this));
    }

    // TODO: replace with LLVM-style casting
    static SymbolExpression* downcast(Expression* s) {
        auto result = dynamic_cast<SymbolExpression*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    String sym_;
};

/**
 * @author JKM
 * @brief An expression to simply reference a variable
 * @details Just serializes the variable's name
 */
class VariableRefExpression : public SymbolExpression {
public:
    /// Ctor for type, var name, and initial value
    VariableRefExpression(const Variable* var)
      : var_(var) {}

    /// Copy ctor
    VariableRefExpression(const VariableRefExpression& other)
      : var_(other.var_) {}

//     Variable* getVariable() {
//         if (!var_)
//             throw_gpusim_exception("No variable set");
//         return var_;
//     }
    const Variable* getVariable() const {
        if (!var_)
            throw_gpusim_exception("No variable set");
        return var_;
    }

    virtual String getSymbol() const { return getVariable()->getName(); }

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new VariableRefExpression(*this));
    }

protected:
    const Variable* var_;
};

/**
 * @author JKM
 * @brief An expression to reference a type
 */
class TypeRefExpression : public Expression {
public:
    /// Ctor
    TypeRefExpression(const Type* tp)
      : tp_(tp) {}

    /// Copy ctor
    TypeRefExpression(const TypeRefExpression& other)
      : tp_(other.tp_) {}

    const Type* getType() const {
        if (!tp_)
            throw_gpusim_exception("No variable set");
        return tp_;
    }

    virtual void serialize(Serializer& s) const {
        s << *tp_;
    }

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new TypeRefExpression(*this));
    }

protected:
    const Type* tp_;
};

/**
 * @author JKM
 * @brief An expression to simply reference a variable
 * @details Just serializes the variable's name
 */
class ArrayIndexExpression : public VariableRefExpression {
public:/**
     * @brief Construct from @ref var and an index expression owning pointer
     */
    ArrayIndexExpression(const Variable* var, ExpressionPtr&& index_exp)
      : VariableRefExpression(var), index_exp_(std::move(index_exp)) {}

    /**
     * @brief Construct from @ref var and an index expression
     * @details Index into an array/object var[$index_exp], where
     * $index_exp will be replace with its corresponding serialized
     * value
     * @param[in] var The array to index into
     * @param[in] index_exp The index expression
     */
    template <class ExpressionT>
    ArrayIndexExpression(const Variable* var, ExpressionT index_exp)
      : ArrayIndexExpression(var, ExpressionPtr(new ExpressionT(std::move(index_exp)))) {}

    /// Copy ctor
    ArrayIndexExpression(const ArrayIndexExpression& other)
      : VariableRefExpression(other), index_exp_(other.index_exp_->clone()) {}

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

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new ArrayIndexExpression(*this));
    }

protected:
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

    /// Copy ctor
    VariableDeclarationExpression(const VariableDeclarationExpression& other)
      : var_(other.var_) {}

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

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new VariableDeclarationExpression(*this));
    }
    // TODO: replace with LLVM-style casting
    static VariableDeclarationExpression* downcast(Expression* s) {
        auto result = dynamic_cast<VariableDeclarationExpression*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    Variable* var_;
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class VariableInitExpression : public VariableDeclarationExpression {
public:
    /// Ctor for var and initial value
    VariableInitExpression(Variable* var, ExpressionPtr&& value_exp)
      : VariableDeclarationExpression(var) {
        value_ = std::move(value_exp);
    }

    template <class InitialValExp>
    /// Ctor for var and initial value
    VariableInitExpression(Variable* var, InitialValExp&& value_exp)
      : VariableDeclarationExpression(var) {
        value_ = ExpressionPtr(new InitialValExp(std::move(value_exp)));
    }

    /// Copy ctor
    VariableInitExpression(const VariableInitExpression& other)
      : VariableDeclarationExpression(other), value_(other.value_->clone()) {}

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

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new VariableInitExpression(*this));
    }

protected:
    ExpressionPtr value_;
};

/**
 * @author JKM
 * @brief References a macro
 */
class MacroExpression : public Expression {
public:
    typedef Type::size_type size_type;
    /**
     * @brief No-arg ctor
     * @note Does not take ownership of @ref mac
     */
    MacroExpression(Macro* mac)
      : mac_(mac) {}

    /// N-arg ctor
    template <class... Args>
    MacroExpression(Macro* mac, Args&&... args)
      :  mac_(mac) {
        passArguments(std::forward<Args>(args)...);
    }

    /// Copy ctor
    MacroExpression(const MacroExpression& other)
      : mac_(other.mac_) {
        for (auto const & arg :  other.args_)
            args_.emplace_back(arg->clone());
    }

    /// Get the macro
    Macro* getMacro() {
        if (!mac_)
            throw_gpusim_exception("No macro set");
        return mac_;
    }
    /// Get the macro
    const Macro* getMacro() const {
        if (!mac_)
            throw_gpusim_exception("No macro set");
        return mac_;
    }

    bool hasMappedArgument(size_type i) const {
        return i < args_.size();
    }

    size_type argCount() const;

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

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new MacroExpression(*this));
    }

protected:
    Macro* mac_;
    typedef std::vector<ExpressionPtr> Args;
    Args args_;
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

    /// Copy ctor
    UnaryExpression(const UnaryExpression& other)
      : operand_(other.operand_->clone()) {}

    // freaking idiotic C++
    // how do they manage to make it such a mess?
    // http://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference
//     template <class OperandExpression, class = typename std::enable_if<!std::is_lvalue_reference<OperandExpression>::value>::type >
//     UnaryExpression(OperandExpression&& operand)
//       : operand_(new OperandExpression(std::move(operand))) {
//     }

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
 * @brief Take reference (usu. address of) of an expression
 * @details e.g. &obj
 */
class ReferenceExpression : public UnaryExpression {
public:
    using UnaryExpression::UnaryExpression;

    template <class OperandExpression, class = typename std::enable_if<!std::is_lvalue_reference<OperandExpression>::value>::type, class = typename std::enable_if<!std::is_same<OperandExpression, ReferenceExpression >::value>::type >
    ReferenceExpression(OperandExpression&& operand)
      : UnaryExpression(ExpressionPtr(new OperandExpression(std::move(operand)))) {
    }

    /// Copy ctor
    ReferenceExpression(const ReferenceExpression& other)
      : UnaryExpression(other) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new ReferenceExpression(*this));
    }
};

/**
 * @author JKM
 * @brief Take reference (usu. address of) of an expression
 * @details e.g. &obj
 */
class UnaryMinusExpression : public UnaryExpression {
public:
    using UnaryExpression::UnaryExpression;

    template <class OperandExpression, class = typename std::enable_if<!std::is_lvalue_reference<OperandExpression>::value>::type, class = typename std::enable_if<!std::is_same<OperandExpression, UnaryMinusExpression >::value>::type >
    UnaryMinusExpression(OperandExpression&& operand)
      : UnaryExpression(ExpressionPtr(new OperandExpression(std::move(operand)))) {
    }

    /// Copy ctor
    UnaryMinusExpression(const UnaryMinusExpression& other)
      : UnaryExpression(other) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new UnaryMinusExpression(*this));
    }
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

    /// Copy ctor
    PreincrementExpression(const PreincrementExpression& other)
      : UnaryExpression(other) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new PreincrementExpression(*this));
    }
};

class PrecedenceExpression : public Expression {
public:
    enum {
        // http://en.cppreference.com/w/cpp/language/operator_precedence
        PREC_GRP1, // ::
        PREC_GRP2,   // suffix ++, --, (), [], fct-style cast
        PREC_GRP3, // prefix ++, --, unary +, -, !, ~, (type), * (indir), &, sizeof, new, new[], delete, delete[]
        PREC_GRP4, // .*   ->*
        PREC_GRP5, // * / %
        PREC_GRP6, // + -
        PREC_GRP7, // << >>
        PREC_GRP8, // < <= > >=
        PREC_GRP15 // ?: = += -= *= /= %= <<= >>= &= ^= |=
    };

    virtual int getPrecedence() const = 0;

    virtual void serialize(Serializer& s) const = 0;

};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class BinaryExpression : public PrecedenceExpression {
public:
    /// Construct from expression pointers
    BinaryExpression(ExpressionPtr&& lhs, ExpressionPtr&& rhs)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    }

    /// Template where both sides are expression rvalues
    template <class LHSExpression, class RHSExpression>
    BinaryExpression(LHSExpression&& lhs, RHSExpression&& rhs)
      : lhs_(new LHSExpression(std::move(lhs))), rhs_(new RHSExpression(std::move(rhs))) {
    }

    /// Template where only the left side is an expression rvalue
    template <class LHSExpression>
    BinaryExpression(LHSExpression&& lhs, ExpressionPtr&& rhs)
      : lhs_(new LHSExpression(std::move(lhs))), rhs_(std::move(rhs)) {
    }

    /// Template where only the right side is an expression rvalue
    template <class RHSExpression>
    BinaryExpression(ExpressionPtr&& lhs, RHSExpression&& rhs)
      : lhs_(std::move(lhs)), rhs_(new RHSExpression(std::move(rhs))) {
    }

    /// Copy ctor
    BinaryExpression(const BinaryExpression& other)
      : lhs_(other.lhs_->clone()), rhs_(other.rhs_->clone()) {}

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
    // to enforce operator precedence
    bool needGroup(const Expression*) const;
    void serializeGroupedExp(Serializer& s, const Expression* e) const;

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

    /// Copy ctor
    AssignmentExpression(const AssignmentExpression& other)
      : BinaryExpression(other) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new AssignmentExpression(*this));
    }

    virtual int getPrecedence() const { return PREC_GRP15; }

    // TODO: replace with LLVM-style casting
    static AssignmentExpression* downcast(Expression* e) {
        auto result = dynamic_cast<AssignmentExpression*>(e);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }
};

/**
 * @author JKM
 * @brief Equality comparison (==)
 */
class EqualityCompExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;

    /// Copy ctor
    EqualityCompExpression(const EqualityCompExpression& other)
      : BinaryExpression(other) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new EqualityCompExpression(*this));
    }

    virtual int getPrecedence() const { return PREC_GRP15; }

    // TODO: replace with LLVM-style casting
    static EqualityCompExpression* downcast(Expression* e) {
        auto result = dynamic_cast<EqualityCompExpression*>(e);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }
};

/**
 * @author JKM
 * @brief Expression to access a member of an object
 * @details Example: @a obj.symbol where @a obj is an object
 * (class, struct) and @a symbol is a member of @a obj
 * @note The RHS expression must be a symbol in a member access
 * expression
 */
class MemberAccessExpression : public BinaryExpression {
public:
    MemberAccessExpression(ExpressionPtr&& lhs, ExpressionPtr&& rhs)
      : BinaryExpression(std::move(lhs), std::move(rhs)) {
        checkRHSIsSymbol();
    }

    template <class LHSExpression>
    MemberAccessExpression(LHSExpression&& lhs, SymbolExpression&& rhs)
      : BinaryExpression(std::move(lhs), std::move(rhs)) {
    }

    /// Copy ctor
    MemberAccessExpression(const MemberAccessExpression& other)
      : BinaryExpression(other) {}

    /**
     * @brief Ctor for left and right expressions
     * @note @a RHSExpression must be convertible to
     * @ref SymbolExpression
     */
    template <class LHSExpression, class RHSExpression>
    MemberAccessExpression(LHSExpression&& lhs, RHSExpression&& rhs)
      : BinaryExpression(std::move(lhs), std::move(rhs)) {
        checkRHSIsSymbol();
    }

    void checkRHSIsSymbol() const {
        SymbolExpression::downcast(rhs_.get());
    }

    virtual int getPrecedence() const { return PREC_GRP4; }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new MemberAccessExpression(*this));
    }
};

/**
 * @author JKM
 * @brief Product expression
 * @details x*y
 */
class ProductExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;

    /// Copy ctor
    ProductExpression(const ProductExpression& other)
      : BinaryExpression(other) {}

    virtual int getPrecedence() const { return PREC_GRP5; }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new ProductExpression(*this));
    }
};

/**
 * @author JKM
 * @brief Sum expression
 * @details x + y
 */
class SumExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;

    /// Copy ctor
    SumExpression(const SumExpression& other)
      : BinaryExpression(other) {}

    virtual int getPrecedence() const { return PREC_GRP6; }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new SumExpression(*this));
    }
};

/**
 * @author JKM
 * @brief Sum expression
 * @details x + y
 */
class SubtractExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;

    /// Copy ctor
    SubtractExpression(const SubtractExpression& other)
      : BinaryExpression(other) {}

    virtual int getPrecedence() const { return PREC_GRP6; }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new SubtractExpression(*this));
    }
};

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class LTComparisonExpression : public BinaryExpression {
public:
    using BinaryExpression::BinaryExpression;
    /// Ctor for lhs & rhs
//     LTComparisonExpression(ExpressionPtr&& lhs, ExpressionPtr&& rhs)
//       : BinaryExpression(std::move(lhs), std::move(rhs)) {
//     }

    /// Copy ctor
    LTComparisonExpression(const LTComparisonExpression& other)
      : BinaryExpression(other) {}

    virtual int getPrecedence() const { return PREC_GRP8; }

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new LTComparisonExpression(*this));
    }
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
class LiteralIntExpression : public LiteralExpression {
public:
    LiteralIntExpression(int i) : i_(i) {}

    /// Copy ctor
    LiteralIntExpression(const LiteralIntExpression& other)
      : i_(other.i_) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new LiteralIntExpression(*this));
    }

protected:
    int i_=0;
};

/**
 * @author JKM
 * @brief A class to encapsulate string literals
 */
class RealLiteralExpression : public LiteralExpression {
public:
    RealLiteralExpression(const double v) : v_(v) {}

    /// Copy ctor
    RealLiteralExpression(const RealLiteralExpression& other)
      : v_(other.v_) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new RealLiteralExpression(*this));
    }

protected:
    double v_;
};

/**
 * @author JKM
 * @brief A class to encapsulate string literals
 */
class StringLiteralExpression : public LiteralExpression {
public:
    StringLiteralExpression(const std::string s) : s_(s) {}

    /// Copy ctor
    StringLiteralExpression(const StringLiteralExpression& other)
      : s_(other.s_) {}

    virtual void serialize(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new StringLiteralExpression(*this));
    }

protected:
    std::string s_;
};

class Function;

/**
 * @author JKM
 * @brief A function call expression
 * @details Represents a call to a function. Expressions
 * may be used to pass parameters.
 * @note Convention: parameters are declared in the function header.
 * @a Arguments are the expressions passed to the function.
 */
class FunctionCallExpression : public Expression {
public:
    typedef Type::size_type size_type;

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

    /// Copy ctor
    FunctionCallExpression(const FunctionCallExpression& other)
      : func_(other.func_) {
        for (auto const &argpair : other.argmap_)
            argmap_.emplace(argpair.first, argpair.second->clone());
        for (auto const &arg : other.extra_args_)
            extra_args_.emplace_back(arg->clone());
    }

    const Function* getFunction() const { return func_; }

    /// Pass the exp @ref v as the arg @ref p of the function
    void passMappedArgument(ExpressionPtr&& v, const FunctionParameter* p);

    bool hasMappedArgument(const FunctionParameter* p) const {
        return argmap_.count(p);
    }

    const Expression* getMappedArgument(const FunctionParameter* p) const {
        if (!hasMappedArgument(p))
            throw_gpusim_exception("No argument mapped to parameter");
        return argmap_.find(p)->second.get();
    }

    const FunctionParameter* getPositionalParam(int i) const;

    const Expression* getPositionalArg(int i) const;

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

    virtual void serializeArgs(Serializer& s) const;

    virtual ExpressionPtr clone() const {
        return ExpressionPtr(new FunctionCallExpression(*this));
    }

    // TODO: replace with LLVM-style casting
    static FunctionCallExpression* downcast(Expression* s) {
        auto result = dynamic_cast<FunctionCallExpression*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

    // TODO: replace with LLVM-style casting
    static const FunctionCallExpression* downcast(const Expression* s) {
        auto result = dynamic_cast<const FunctionCallExpression*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

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

/**
 * @author JKM
 * @brief A statement containing a typedef
 */
class BreakStatement : public Statement {
public:
    /// Ctor for typedef @a target @a alias
    BreakStatement() {}

    virtual void serialize(Serializer& s) const;

    static BreakStatement* downcast(Statement* s) {
        BreakStatement* result = dynamic_cast<BreakStatement*>(s);
        if (!result)
            throw_gpusim_exception("Not a BreakStatement");
        return result;
    }
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

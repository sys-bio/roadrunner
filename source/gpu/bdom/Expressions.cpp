// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Expressions.cpp
 *
 *  Created on: Aug 22, 2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "Expressions.hpp"
# include "Structures.hpp"

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

// -- Variable --

void Variable::serialize(Serializer& s) const {
    type_->serializeFirstPart(s);
    s << String(" ");
    s << name_;
    type_->serializeSecondPart(s);
}

// -- SymbolExpression --

void SymbolExpression::serialize(Serializer& s) const {
    s << getSymbol();
}

// -- ArrayIndexExpression --

void ArrayIndexExpression::serialize(Serializer& s) const {
    VariableRefExpression::serialize(s);
    s << "[" << *getIndexExp()<< "]";
}

// -- VariableDeclarationExpression --

void VariableDeclarationExpression::serialize(Serializer& s) const {
    s << *getVariable();
}

// -- VariableInitExpression --

void VariableInitExpression::serialize(Serializer& s) const {
    VariableDeclarationExpression::serialize(s);
    s << " = ";
    s << *getValue();
}

// -- MacroExpression --

MacroExpression::size_type MacroExpression::argCount() const {
    return mac_->argCount();
}

bool MacroExpression::isVarargs() const {
    return mac_->isVarargs();
}

void MacroExpression::passArgument(ExpressionPtr&& v) {
    if (args_.size() >= mac_->argCount())
        throw_gpusim_exception("Macro overflow: cannot pass argument");
    args_.emplace_back(std::move(v));
}

void MacroExpression::serialize(Serializer& s) const {
    s << getMacro()->getName();
    if (args_.size()) {
        s << "(";
        int n=0;
        for (auto const &a :  args_)
            s << (n++ ? ", " : "") << *a;
        s << ")";
    }
}

// -- PreincrementExpression --

void PreincrementExpression::serialize(Serializer& s) const {
    s << "++" << *getOperand();
}

// -- ReferenceExpression --

void ReferenceExpression::serialize(Serializer& s) const {
    s << "&" << *getOperand();
}

// -- UnaryMinusExpression --

void UnaryMinusExpression::serialize(Serializer& s) const {
    s << "-" << *getOperand();
}

// -- BinaryExpression --

bool BinaryExpression::needGroup(const Expression* e) const {
    if (const PrecedenceExpression* x = dynamic_cast<const PrecedenceExpression*>(e))
        if (x->getPrecedence() > getPrecedence())
            return true;
    return false;
}

void BinaryExpression::serializeGroupedExp(Serializer& s, const Expression* e) const {
    if (needGroup(e))
        s << "(" << *e << ")";
    else
        s << *e;
}

// -- AssignmentExpression --

void AssignmentExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << " = ";
    serializeGroupedExp(s, getRHS());
}

// -- EqualityCompExpression --

void EqualityCompExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << " == ";
    serializeGroupedExp(s, getRHS());
}

// -- MemberAccessExpression --

void MemberAccessExpression::serialize(Serializer& s) const {
    checkRHSIsSymbol();
    serializeGroupedExp(s, getLHS());
    s << ".";
    serializeGroupedExp(s, getRHS());
}

// -- ProductExpression --

void ProductExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << "*";
    serializeGroupedExp(s, getRHS());
}

// -- SumExpression --

void SumExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << " + ";
    serializeGroupedExp(s, getRHS());
}

// -- SubtractExpression --

void SubtractExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << " + ";
    serializeGroupedExp(s, getRHS());
}

// -- LTComparisonExpression --

void LTComparisonExpression::serialize(Serializer& s) const {
    serializeGroupedExp(s, getLHS());
    s << " < ";
    serializeGroupedExp(s, getRHS());
}

// -- FunctionCallExpression --

void FunctionCallExpression::passMappedArgument(ExpressionPtr&& v, const FunctionParameter* p) {
    if (!argmap_.count(p)) {
        argmap_.emplace(p, std::move(v));
    } else
        throw_gpusim_exception("Parameter already mapped");
}

const FunctionParameter* FunctionCallExpression::getPositionalParam(int i) const {
    return func_->getPositionalParam(i);
}

const Expression* FunctionCallExpression::getPositionalArg(int i) const {
    const FunctionParameter* param = getPositionalParam(i);
    return getMappedArgument(param);
}

FunctionCallExpression::size_type FunctionCallExpression::getNumPositionalParams() const {
    return func_->getNumPositionalParams();
}

bool FunctionCallExpression::isVarargs() const {
    return func_->isVarargs();
}

void FunctionCallExpression::passArgument(ExpressionPtr&& v) {
    for(size_type i=0; i<getNumPositionalParams(); ++i) {
        if (!hasMappedArgument(getPositionalParam(i))) {
            passMappedArgument(std::move(v), getPositionalParam(i));
            return;
        }
    }
    if (isVarargs()) {
//         Log(Logger::LOG_DEBUG) << "Passing extra argument to " << getFunction()->getName() << "\n";
        passExtraArg(std::move(v));
        return;
    }
    throw_gpusim_exception("Cannot pass argument: out of parameters");
}

void FunctionCallExpression::serialize(Serializer& s) const {
    if (func_->requiresSpecialCallingConvention())
        throw_gpusim_exception("Function call requires special calling convention; cannot serialize");
    s << func_->getName() << "(";
    serializeArgs(s);
    s << ")";
}

void FunctionCallExpression::serializeArgs(Serializer& s) const {
    int n=0;
    for (n=0; n<(int)getNumPositionalParams();++n) {
        s << (n ? ", " : "") << *getMappedArgument(getPositionalParam(n));
    }
    if (isVarargs()) {
        for (auto const &p : extra_args_) {
            s << (n++ ? ", " : "") << *p;
        }
    } else if(extra_args_.size())
        throw_gpusim_exception("Sanity check failed: have extra args but function is not marked variadic");
}

// -- LiteralIntExpression --

void LiteralIntExpression::serialize(Serializer& s) const {
    s << i_;
}

// -- RealLiteralExpression --

void RealLiteralExpression::serialize(Serializer& s) const {
    s << v_;
}

// -- StringLiteralExpression --

void StringLiteralExpression::serialize(Serializer& s) const {
    s << "\"" << s_ << "\"";
}

// -- StatementContainer --

void StatementContainer::serialize(Serializer& s) const {
    for (Statement* m : getStatements()) {
        m->serialize(s);
    }
}

Statement* StatementContainer::addStatement(StatementPtr&& stmt) {
    stmts_.emplace_back(std::move(stmt));
    return stmts_.back().get();
}

Statement* StatementContainer::addStatement(ExpressionPtr&& exp) {
    return addStatement(StatementPtr(new ExpressionStatement(std::move(exp))));
}

// -- ExpressionStatement --

void ExpressionStatement::serialize(Serializer& s) const {
    getExpression()->serialize(s);
    s << ";" << nl;
}

// -- TypedefStatement --

TypedefStatement::TypedefStatement(Type* target, const std::string& alias)
  : target_(target), alias_(BaseTypes::get().newAliasType(target, alias)) {

}

void TypedefStatement::serialize(Serializer& s) const {
    s << "typedef ";
    s << *getTarget() << " " << *getAlias();
    s << ";" << nl;
}

// -- BreakStatement --

void BreakStatement::serialize(Serializer& s) const {
    s << "break;" << nl;
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

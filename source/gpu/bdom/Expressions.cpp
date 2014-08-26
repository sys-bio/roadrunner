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

// -- VariableRefExpression --

void VariableRefExpression::serialize(Serializer& s) const {
    s << getVariable()->getName();
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

void MacroExpression::serialize(Serializer& s) const {
    s << getMacro()->getName();
}

// -- PreincrementExpression --

void PreincrementExpression::serialize(Serializer& s) const {
    s << "++" << *getOperand();
}

// -- AssignmentExpression --

void AssignmentExpression::serialize(Serializer& s) const {
    s << *getLHS();
    s << " = ";
    s << *getRHS();
}

// -- LTComparisonExpression --

void LTComparisonExpression::serialize(Serializer& s) const {
    s << *getLHS();
    s << " < ";
    s << *getRHS();
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
        passExtraArg(std::move(v));
        return;
    }
    throw_gpusim_exception("Cannot pass argument: out of parameters");
}

void FunctionCallExpression::serialize(Serializer& s) const {
    if (func_->requiresSpecialCallingConvention())
        throw_gpusim_exception("Function call requires special calling convention; cannot serialize");
    s << func_->getName() << "(";
    int n=0;
    for (auto const &p : argmap_) {
        s << (n++ ? ", " : "") << *p.second;
    }
    if (isVarargs()) {
        for (auto const &p : extra_args_) {
            s << (n++ ? ", " : "") << *p;
        }
    } else if(extra_args_.size())
        throw_gpusim_exception("Sanity check failed: have extra args but function is not marked variadic");
    s << ")";
}

// -- LiteralIntExpression --

void LiteralIntExpression::serialize(Serializer& s) const {
    s << i_;
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

} // namespace dom

} // namespace rrgpu

} // namespace rr

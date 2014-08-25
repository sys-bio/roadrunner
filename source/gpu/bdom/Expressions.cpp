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

// FunctionCallExpression::FunctionCallExpression(const Function* func, std::initializer_list<Expression*> args) {
//
// }

void VariableRefExpression::serialize(Serializer& s) const {
    s << getVariable()->getName();
}

void VariableDeclarationExpression::serialize(Serializer& s) const {
    s << *getVariable();
}

void VariableInitExpression::serialize(Serializer& s) const {
    VariableDeclarationExpression::serialize(s);
    s << " = ";
    s << *getValue();
}

void MacroExpression::serialize(Serializer& s) const {
    s << getMacro()->getName();
}

void PreincrementExpression::serialize(Serializer& s) const {
    s << "++" << *getOperand();
}

void LTComparisonExpression::serialize(Serializer& s) const {
    s << *getLHS();
    s << " < ";
    s << *getRHS();
}

void FunctionCallExpression::passMappedArgument(ExpressionPtr&& v, const FunctionParameter* p) {
    if (!argmap_.count(p)) {
        argmap_.emplace(p, std::move(v));
    } else
        throw_gpusim_exception("Parameter already mapped");
}

const FunctionParameter* FunctionCallExpression::getPositionalParam(int i) const {
    return func_->getPositionalParam(i);
}

void FunctionCallExpression::serialize(Serializer& s) const {
    if (func_->requiresSpecialCallingConvention())
        throw_gpusim_exception("Function call requires special calling convention; cannot serialize");
    s << func_->getName() << "(";
    for (auto const &p : argmap_) {
        s << *p.second;
    }
    s << ")";
}

void LiteralIntExpression::serialize(Serializer& s) const {
    s << i_;
}

void StringLiteralExpression::serialize(Serializer& s) const {
    s << "\"" << s_ << "\"";
}

void ExpressionStatement::serialize(Serializer& s) const {
    getExpression()->serialize(s);
    s << ";" << nl;
}

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

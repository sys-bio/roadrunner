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

void FunctionCallExpression::passArgument(FunctionParameter* p, Variable* v) {
    if (!argmap_.count(p)) {
        argmap_.emplace(p, v);
    } else
        throw_gpusim_exception("Parameter already mapped");
}

void FunctionCallExpression::serialize(Serializer& s) const {
    s << func_->getName() << "(";
    for (auto p : argmap_) {
        s << p.second->getName();
    }
    s << ")";
}

void ExpressionStatement::serialize(Serializer& s) const {
    getExpression()->serialize(s);
    s << ";" << nl;
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

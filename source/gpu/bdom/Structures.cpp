// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Structures.cpp
 *
 *  Created on: Aug 21, 2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "Structures.hpp"

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

void Block::serialize(Serializer& s) const {
    s << "{";
    {
        IndentationBumper b(s);
        s << nl;
        StatementContainer::serialize(s);
    }

    s << "}" << nl;
}

void Function::serialize(Serializer& s) const {
    // serialize the header
    serializeHeader(s);

    // serialize the body
    Block::serialize(s);
}

void Function::serializeHeader(Serializer& s) const {
    if (hasCLinkage())
        s << "extern \"C\" ";

    returnTp_->serialize(s);
    s << " ";
    s << name_;
    s << "(";

    int n=0;
    for (const Variable* arg : getArgs()) {
        s << (n++ ? ", " : "");
        arg->serialize(s);
    }

    if (isVarargs()) {
        s << (n++ ? ", " : "");
        s << "...";
    }

    s << ") ";
}

// -- Module --

void Module::serializeMacros(Serializer& s) const {
    for (Macro* m : getMacros()) {
        m->serialize(s);
    }
}

// -- ForStatement --

ForStatement::ForStatement() {

}

ForStatement::ForStatement(ExpressionPtr&& init_exp, ExpressionPtr&& cond_exp, ExpressionPtr&& loop_exp)
  : init_exp_(std::move(init_exp)),
  cond_exp_(std::move(cond_exp)),
  loop_exp_(std::move(loop_exp))
  {

}

void ForStatement::serialize(Serializer& s) const {
    s << "for (" << *getInitExp() << ";" << *getCondExp() << ";" << *getLoopExp() << ") ";
    s << getBody();
}

// -- CaseLabel --

void CaseLabel::serialize(Serializer& s) const {
    s << "case " << *getValue() << ":" << nl;
}

// -- CaseLabel --

void DefaultLabel::serialize(Serializer& s) const {
    s << "default:" << nl;
}

// -- SwitchStatement --

SwitchStatement::SwitchStatement(ExpressionPtr&& exp)
  : exp_(std::move(exp)) {

}

void SwitchStatement::serialize(Serializer& s) const {
    s << "switch (" << *getExpression() << ")";
    s << getBody();
}


} // namespace dom

} // namespace rrgpu

} // namespace rr

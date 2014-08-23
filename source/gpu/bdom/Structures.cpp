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

Block::~Block() {}

void Block::serialize(Serializer& s) const {
    s << "{";
    {
        IndentationBumper b(s);
        s << nl;

        for (const Statement* t : getStatements()) {
            t->serialize(s);
    //         s << nl;
        }
    }

    s << "}" << nl;
}

void Function::serialize(Serializer& s) const {
    // serialize the header
    returnTp_->serialize(s);
    s << " ";
    s << name_;
    s << "(";

    int n=0;
    for (const Variable* arg : getArgs()) {
        s << (n ? ", " : "");
        arg->serialize(s);
    }

    s << ") ";

    // serialize the body
    Block::serialize(s);
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

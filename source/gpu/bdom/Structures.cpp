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

void Function::serialize(std::ostream& os) const {
    // serialize the header
    returnTp_->serialize(os);
    os << " ";
    os << name_;
    os << "(";

    int n=0;
    for (const Variable* arg : getArgs()) {
        os << (n ? ", " : "");
        arg->serialize(os);
    }
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

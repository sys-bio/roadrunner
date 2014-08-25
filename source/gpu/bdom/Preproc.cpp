// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Preproc.cpp
 *
 *  Created on: Aug 24, 2014
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

void Macro::serialize(Serializer& s) const {
    s << "#define " << getName();
    if (argCount()) {
        s << "(";
        int n=0;
        for (MacroArg* arg : getArgs()) {
            if (n++)
                s << ", ";
            s << *arg;
        }
        s << ")";
    }
    s << " " << getContent() << "\n";
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

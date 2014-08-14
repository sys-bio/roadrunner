// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Range.hpp
 *
 *  Created on: Aug 14, 2014
 *      Author: JKM
 */

#ifndef rrRangeH
#define rrRangeH

// == INCLUDES ================================================

#include <type_traits>

// == CODE ====================================================

namespace rr
{

/**
 * @author JKM
 * @brief Allows iterating over a range,
 * defined by begin and end iterators
 */
template <class Iterator>
class Range {
public:
    typedef Iterator IteratorType;

    /// Construct from begin/end iterators
    Range(IteratorType begin, IteratorType end)
        : begin_(begin), end_(end) {}

    /// Construct from container
    template<class Contiainer>
    Range(Contiainer& c)
        : begin_(c.begin()), end_(c.end()) {}

    /// Construct from container
    template<class Contiainer>
    Range(const Contiainer& c)
        : begin_(c.begin()), end_(c.end()) {}

    /// Required begin func
    IteratorType begin() { return begin_; }
    /// Required end func
    IteratorType end() { return end_; }

protected:
    IteratorType begin_, end_;
};

} // namespace rr

#endif // rrAccessPtrIterH

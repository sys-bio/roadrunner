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
 * @details Use this class with a custom type that defines
 * non-standard begin/end functions. The result is an object
 * with the usual begin/end iterator semantics, which enables
 * use within a range-based for loop, among other things.
 * @verbat
 * // example usage
 * class graph {
 *     node_iterator nodes_begin();
 *     node_iterator nodes_end();\n
 *     // return a range of all nodes
 *     Range<graph::node_iterator> nodes() {
 *         return Range<graph::node_iterator>(g.nodes_begin(), g.nodes_end());
 *     }\n
 *     edge_iterator edges_begin();
 *     edge_iterator edges_end();\n
 *     // return a range of all edges
 *     Range<graph::edge_iterator> edges() {
 *         return Range<graph::edge_iterator>(g.edges_begin(), g.edges_end());
 *     }\n
 * };\n
 * // type graph has no begin/end functions and therefore cannot
 * // be used in a range-based for, but its range methods can
 * graph g;\n
 * // loop over nodes
 * for (node n : g.nodes())
 *     std::cout << n << std::endl;\n
 * // loop over edges
 * for (edge e : g.edges)
 *     std::cout << e << std::endl;
 * @endverbat
 */
template <class Iterator>
class Range {
public:
    typedef Iterator IteratorType;
    typedef Iterator iterator;

    /// Construct from begin/end iterators
    Range(IteratorType begin, IteratorType end)
        : begin_(begin), end_(end) {}

    /// Construct from container with begin/end functions
    template<class Contiainer>
    Range(Contiainer& c)
        : begin_(c.begin()), end_(c.end()) {}

    /// Construct from container with begin/end functions
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

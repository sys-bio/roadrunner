// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * AccessPtrIterator.hpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

#ifndef rrAccessPtrIterH
#define rrAccessPtrIterH

// == INCLUDES ================================================

#include <type_traits>

// == CODE ====================================================

namespace rr
{


/**
 * @author JKM
 * @brief Iterates over access pointers (i.e. raw pointers)
 * for an owning pointer (e.g. std::unique_ptr)
 * @details Usually, an object that owns a resource does not
 * expose the owning pointer externally. Instead, use this class,
 * especially with containers of owning pointers, to retrieve access
 * pointers.
 */
template <class Iterator>
class AccessPtrIterator {
public:
    typedef Iterator difference_type;
    typedef typename Iterator::value_type::pointer value_type;
    typedef value_type    pointer;
    typedef std::add_lvalue_reference<value_type>  reference;
    typedef std::output_iterator_tag iterator_category; // what's this for?

    AccessPtrIterator(Iterator i)
        : i_(i) {}

    AccessPtrIterator& operator++() {
        ++i_;
        return *this;
    }

    /// Returns the value directly (no need for double indirection)
    pointer operator->() {
        return i_->get();
    }

    pointer operator*() {
        return i_->get();
    }

    bool operator!=(const AccessPtrIterator& other) const {
        return i_ != other.i_;
    }

protected:
    Iterator i_;
};

} // namespace rr

#endif // rrAccessPtrIterH

// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/Preproc.hpp
  * @author JKM
  * @date 08/24/2014
  * @copyright Apache License, Version 2.0
  * @brief Preprocessor elements
  * @details Macros etc.
**/

#ifndef rrGPU_BDOM_Preproc_H
#define rrGPU_BDOM_Preproc_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include "BaseTypes.hpp"
# include "Expressions.hpp"

# include <iostream>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

/**
 * @author JKM
 * @brief A preprocessor macro
 */
class Macro {
public:
    /// Ctor with no args
    Macro(const String& name)
      : name_(name) {}

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

protected:
    std::string name_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

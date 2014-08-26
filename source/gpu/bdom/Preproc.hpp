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
# include "patterns/AccessPtrIterator.hpp"

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
class MacroArg {
public:
    typedef BDOM_String String;
    MacroArg(const String& name)
      : name_(name) {}

    ~MacroArg() {}

    virtual void serialize(Serializer& s) const {
        s << getName();
    }

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

protected:
    String name_;
};
typedef DomOwningPtr<MacroArg> MacroArgPtr;

inline Serializer& operator<<(Serializer& s,  const MacroArg& a) {
    a.serialize(s);
    return s;
}

/**
 * @author JKM
 * @brief A preprocessor macro
 */
class Macro {
protected:
    typedef std::vector<MacroArgPtr> Args;
public:
    typedef MacroArg::String String;
    typedef std::size_t size_type;

    /// Ctor with no args
    Macro(const String& name, const String& content)
      : name_(name), content_(content) {}

    /// Ctor with args
    template <typename ...Args>
    Macro(const String& name, const String& content, Args... args)
      : name_(name), content_(content) {
        addArgs(args...);
    }

    template <typename Arg, typename ...Args>
    void addArgs(Arg arg, Args... rest) {
        addArg(arg);
        addArgs(rest...);
    }

    template <typename Arg>
    void addArgs(Arg arg) {
        addArg(arg);
    }

    void addArg(const String& name) {
        args_.emplace_back(new MacroArg(name));
    }

    bool isVarargs() const {
        return is_varargs_;
    }

    void setIsVarargs(bool val) {
        is_varargs_ = val;
    }

    const String& getName() const { return name_; }
    void setName(const String& name) { name_ = name; }

    const String& getContent() const { return content_; }
    void setContent(const String& content) { content_ = content; }

    size_type argCount() const { return args_.size(); }

    virtual void serialize(Serializer& s) const;

    typedef AccessPtrIterator<Args::iterator> ArgIterator;
    typedef AccessPtrIterator<Args::const_iterator> ConstArgIterator;

    typedef Range<ArgIterator> ArgRange;
    typedef Range<ConstArgIterator> ConstArgRange;

    ArgRange getArgs() { return ArgRange(args_); }
    ConstArgRange getArgs() const { return ConstArgRange(args_); }

protected:
    String name_;
    String content_;
    Args args_;
    bool is_varargs_ = false;
};
typedef DomOwningPtr<Macro> MacroPtr;

inline Serializer& operator<<(Serializer& s, const Macro& m) {
    m.serialize(s);
    return s;
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

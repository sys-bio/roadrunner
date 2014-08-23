// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file gpu/bdom/Serializer.h
  * @author JKM
  * @date 08/22/2014
  * @copyright Apache License, Version 2.0
  * @brief Serializer for the DOM
**/

#ifndef rrGPU_BDOM_Serializer_H
#define rrGPU_BDOM_Serializer_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include <fstream>
# include <typeinfo>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

template <
    class CharT,
    class Traits = std::char_traits<CharT>
    >
class SerializerT : public std::basic_ofstream<CharT, Traits> {
public:
    using std::basic_ofstream<CharT, Traits>::basic_ofstream;

    void changeIndentation(int amount) {
        ind_ += amount;
    }

    void newline() {
        *this << "\n";
        for (int i=0; i<ind_; ++i)
            *this << " ";
    }

protected:
    int ind_;
};

typedef SerializerT<char> Serializer;

class SerializerNewline {

};

inline std::ostream& operator<<(std::ostream& os,  const SerializerNewline& nl) {
    try {
        Serializer& s(dynamic_cast<Serializer&>(os));
        s.newline();
    } catch (std::bad_cast) {
        os << "\n";
    }
    return os;
}

extern SerializerNewline nl;

class SerializerOptionBumper {
public:
};

class IndentationBumper : public SerializerOptionBumper {
public:
    IndentationBumper(Serializer& s)
      :  s_(s) {
        s_.changeIndentation(2);
    }
    ~IndentationBumper() {
        s_.changeIndentation(-2);
    }

protected:
    Serializer& s_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

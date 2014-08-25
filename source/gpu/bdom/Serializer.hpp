// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file gpu/bdom/Serializer.hpp
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

# include <iostream>
# include <fstream>
# include <typeinfo>
# include <memory>
# include <cassert>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

typedef std::string BDOM_String;

// template <
//     class CharT,
//     class Traits = std::char_traits<CharT>
//     >
// class SerializerT : public std::basic_ostream<CharT, Traits> {
// public:
//     typedef BDOM_String String;
//     using std::basic_ostream<CharT, Traits>::basic_ostream;
//
//     SerializerT(const String& file)
//       :
// //         sbuf_(new std::basic_filebuf<CharT, Traits>()),
//         SerializerT(new std::basic_filebuf<char>()) {
// //         this->init(sbuf_.get());
//         std::basic_filebuf<char>* fb = dynamic_cast<std::basic_filebuf<char>*>(this->rdbuf());
//         assert(fb);
//         sbuf_.reset(fb);
//         sbuf_->open(file, std::ios_base::out|std::ios_base::trunc);
// //         *this << "xyz\n";
// //         sbuf_->close();
//     }
//
// //     SerializerT(const String& file) {
// //     }
//
//     void changeIndentation(int amount) {
//         ind_ += amount;
//     }
//
//     void newline() {
//         *this << "\n";
//         std::cerr << "ind_ = " << ind_ << "\n";
//         for (int i=0; i<ind_; ++i)
//             *this << " ";
//     }
//
// protected:
//     int ind_=0;
//     std::unique_ptr<std::basic_filebuf<CharT, Traits>> sbuf_;
// };

template <
    class CharT,
    class Traits = std::char_traits<CharT>
    >
class SerializerT {
protected:
    typedef std::unique_ptr<std::ostream> OStreamPtr;
public:
    typedef BDOM_String String;

    SerializerT(const String& file)
      : os_(new std::ofstream(file)) {

    }

    SerializerT& operator<<(const std::string& str) {
        doIndent();
        assert(os_);
        *os_ << str;
        return *this;
    }

    SerializerT& operator<<(int i) {
        doIndent();
        assert(os_);
        *os_ << i;
        return *this;
    }

    void changeIndentation(int amount) {
        ind_ += amount;
    }

    void newline() {
        *this << "\n";
        nlflag_ = true;
//         std::cerr << "ind_ = " << ind_ << "\n";
    }

protected:
    void doIndent() {
        if (nlflag_) {
            assert(os_);
            for (int i=0; i<ind_; ++i)
                *os_ << " ";
            nlflag_ = false;
        }
    }

    int ind_=0;
    /// True when the cursor is positioned at a new line
    bool nlflag_=true;
    OStreamPtr os_;
};

typedef SerializerT<char> Serializer;

class SerializerNewline {

};

inline Serializer& operator<<(Serializer& s,  const SerializerNewline& nl) {
    s.newline();
    return s;
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

class SerializableElement {
public:
    virtual void serialize(Serializer& s) const = 0;
};

inline Serializer& operator<<(Serializer& s, const SerializableElement& e) {
    e.serialize(s);
    return s;
}

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard

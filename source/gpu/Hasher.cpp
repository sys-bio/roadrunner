// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README


/*
 * Hasher.cpp
 *
 *  Created on: 09/07/2014
 *
 * Author: JKM
 */

// == INCLUDES ================================================

# include "Hasher.hpp"
# include <sstream>
# include <iomanip>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

// -- Hashval --

std::string Hashval::str() const {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) <<
      digest_[0] << digest_[1] << digest_[2] << digest_[3] << digest_[4];
    return ss.str();
}

Hashval Hashval::combined(const Hashval& other) {
    Hashval result;
    // Based on impl from boost::hash which is in turn based on Hoad & Zobel
    // http://goanna.cs.rmit.edu.au/~jz/fulltext/jasist-tch.pdf
    // ref: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
    result.digest_[0] = digest_[0] + 0x9e3779b9 + (other.digest_[0]<<6) + (other.digest_[0]>>2);
    result.digest_[1] = digest_[1] + 0x9e3779b9 + (other.digest_[1]<<6) + (other.digest_[1]>>2);
    result.digest_[2] = digest_[2] + 0x9e3779b9 + (other.digest_[2]<<6) + (other.digest_[2]>>2);
    result.digest_[3] = digest_[3] + 0x9e3779b9 + (other.digest_[3]<<6) + (other.digest_[3]>>2);

    return result;
}

Hashval::Hashval(sha1::SHA1& s) {
    useDigest(s);
}

void Hashval::useDigest(sha1::SHA1& s) {
    s.getDigest(digest_);
}

// -- Hash --

Hashval Hash::me(const std::string& str) {
    sha1::SHA1 sha;
    sha.processBytes(str.c_str(), str.size());
    Hashval::digest32_t d;
    return Hashval(sha);
}

} // namespace rrgpu

} // namespace rr

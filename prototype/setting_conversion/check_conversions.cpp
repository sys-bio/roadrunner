#include "Setting.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <climits>
#include <cfloat>
#include <iomanip>
#include <locale>
#include <set>
#include <utility>

// Only these two out-of-line definitions are used, copied from source/Setting.cpp.
namespace rr {
Setting::Setting(setting_t value) : value_(std::move(setting_t(std::move(value)))) {}
const std::type_info &Setting::typeInfo() const {
    return std::visit([](auto &&x) -> decltype(auto) {
        return std::move(typeid(x));
    }, value_);
}
}

using Source = std::pair<std::string, rr::setting_t>;
std::vector<Source> sources;
std::set<std::string> sourceKeys;
std::string localeName;
bool probeNative = false;
std::string hex(const void* data, std::size_t n) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (auto p = static_cast<const unsigned char*>(data); n-- > 0; ++p) {
        out += digits[*p >> 4];
        out += digits[*p & 15];
    }
    return out;
}
std::string encode(std::monostate) { return "_"; }
std::string encode(const std::string& s) { return hex(s.data(), s.size()); }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
std::string encode(T v) {
    // x86-64 Linux uses the 80-bit x87 value and 6 indeterminate padding bytes.
    if constexpr (std::is_same_v<T,long double> && LDBL_MANT_DIG == 64 &&
                  sizeof(long double) == 16) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return hex(&v, 10);
#else
#error Review the extended-precision representation on this architecture.
#endif
    }
    return hex(&v, sizeof(v));
}
template<class T>
std::string encode(const std::vector<T>& values) {
    std::string s = "[";
    for (const auto& v : values) { s += encode(v); s += ","; }
    return s + "]";
}
template<class T> void add(T val, std::string label) {
    std::string key = std::string(typeid(T).name()) + ":" + encode(val);
    if (sourceKeys.insert(key).second) sources.emplace_back(key + ":" + label, rr::setting_t(val));
}
template<class T> void integers() {
    using L = std::numeric_limits<T>;
    add(T(0),"zero"); add(T(1),"one"); add(T(2),"two"); add(T(42),"positive");
    add(T(L::lowest()),"min"); add(T(L::lowest() + 1),"min-plus-one");
    add(T(L::max()),"max"); add(T(L::max() - 1),"max-minus-one");
    if constexpr (std::is_signed_v<T>) { add(T(-1),"minus-one"); add(T(-42),"negative"); }
    for (int exponent : {7,8,15,16,24,31,32,53,63}) {
        std::uint64_t pivot = UINT64_C(1) << exponent;
        for (int offset : {-1,0,1}) {
            auto value = pivot + offset;
            if (value <= static_cast<std::uint64_t>(L::max()))
                add(static_cast<T>(value),"power-boundary");
            if constexpr (std::is_signed_v<T>)
                if (value <= static_cast<std::uint64_t>(L::max()))
                    add(static_cast<T>(-static_cast<std::int64_t>(value)),"negative-power-boundary");
        }
    }
}
template<class T> void floats() {
    using L=std::numeric_limits<T>;
    add(T(0),"+0"); add(-T(0),"-0");
    for (T val : {T(1),T(0.25),T(0.75),T(1.5),T(42),L::denorm_min(),L::min(),L::max(),
                  L::infinity(),L::quiet_NaN()}) {
        add(val,"positive"); add(T(-val),"negative");
    }
    // Quiet NaN with a second payload, plus the negative bit pattern.
    T nanPayload;
    if constexpr (sizeof(T)==4) { std::uint32_t s=0x7fc05555u; std::memcpy(&nanPayload,&s,4); }
    else { std::uint64_t s=UINT64_C(0x7ff8000000005555); std::memcpy(&nanPayload,&s,8); }
    add(nanPayload,"nan-payload"); add(T(-nanPayload),"negative-nan-payload");
    for (int exponent : {7,8,15,16,24,31,32,53,63,64}) {
        T p=std::ldexp(T(1),exponent);
        for(T val : {std::nextafter(p,T(0)),p,std::nextafter(p,L::infinity())}) {
            add(val,"integral-boundary"); add(T(-val),"negative-integral-boundary");
        }
    }
    if constexpr (std::is_same_v<T,double>) {
        double p=std::numeric_limits<float>::max();
        add(p,"float-max"); add(-p,"negative-float-max");
        add(std::nextafter(p,L::infinity()),"next-after-float-max");
        add(T(2)*p,"above-float-max");
    }
}
template<class D> const char* classification(const rr::setting_t& src) {
    return std::visit([](auto const& x) {
        using S=std::decay_t<decltype(x)>;
        // The original code throws before the numeric cast for stored float +inf.
        if constexpr (std::is_same_v<S,float> && std::is_arithmetic_v<D>)
            if (x > (std::numeric_limits<float>::max)()) return "value";
        if constexpr (std::is_floating_point_v<S> && std::is_integral_v<D> && !std::is_same_v<D,bool>) {
            long double n=std::trunc(static_cast<long double>(x));
            long double lim=std::ldexp(1.L,std::numeric_limits<D>::digits);
            long double bottom=std::is_signed_v<D> ? -lim : 0.L;
            if (!(bottom<=n && n<lim)) return "undefined-fp-int";
        } else if constexpr(std::is_same_v<S,double> && std::is_same_v<D,float>) {
            if (std::isfinite(x) && std::abs(x)>static_cast<double>((std::numeric_limits<float>::max)()))
                return "native-fp-narrow";
        } else if constexpr (std::is_integral_v<S> && std::is_integral_v<D> && std::is_signed_v<D>) {
            if constexpr (std::is_signed_v<S>) {
                if (static_cast<std::intmax_t>(x) < (std::numeric_limits<D>::lowest)() ||
                    static_cast<std::intmax_t>(x) > (std::numeric_limits<D>::max)()) return "impl-defined-int";
            } else {
                if (static_cast<std::uintmax_t>(x) > static_cast<std::uintmax_t>((std::numeric_limits<D>::max)()))
                    return "impl-defined-int";
            }
        }
        return "value";
    }, src);
}
template<class T> T implicit(const rr::Setting& setting) { return setting; }
template<class D> void check(const rr::Setting& setting, const Source& source) {
    std::string kind = classification<D>(source.second);
    bool native = kind == "undefined-fp-int" || kind == "native-fp-narrow";
    if (native != probeNative) return;
    for (auto mode : {0,1}) {
        std::cout << localeName << '\t' << source.first << '\t' << typeid(D).name() << '\t'
                  << (mode ? "implicit" : "getAs") << '\t';
        try {
            D result = mode ? implicit<D>(setting) : setting.getAs<D>();
            std::cout << kind << '\t' << encode(result);
        } catch(std::exception const& ex) {
            std::cout << "exception" << '\t' << typeid(ex).name() << ':' << encode(std::string(ex.what()));
        }
        std::cout << '\n';
    }
}
struct Grouped : std::numpunct<char> {
    std::string do_grouping() const override { return "\3"; }
    char do_thousands_sep() const override { return '_'; }
};
int main(int argc, char** argv){
    probeNative = argc > 1 && std::string(argv[1]) == "--native-observations";
    add(std::monostate{},"empty");
    add(std::string(),"empty-string"); add(std::string("a\0b",3),"embedded-nul"); add(std::string("-17.5"),"text");
    add(false,"false"); add(true,"true");
    integers<std::int32_t>(); integers<std::uint32_t>(); integers<std::int64_t>(); integers<std::uint64_t>();
    integers<char>(); integers<unsigned char>(); floats<float>(); floats<double>();
    add(std::vector<double>{},"empty-vector"); add(std::vector<double>{-0.,1.,INFINITY,NAN},"special-vector");
    add(std::vector<std::string>{},"empty-string-vector");
    add(std::vector<std::string>{"",std::string("a\0b",3),"xyz"},"string-vector");
    for (std::string locale : {"classic","grouped"}) {
        localeName=locale;
        std::locale::global(locale=="classic" ? std::locale::classic() : std::locale(std::locale::classic(),new Grouped));
        for (auto const& source : sources) {
            rr::Setting s(source.second);
            // Exactly all 13 storable alternatives, plus the other fundamental
            // C++17 arithmetic types on Linux LP64 (int64_t=long).
            check<std::monostate>(s,source); check<std::string>(s,source); check<bool>(s,source);
            check<std::int32_t>(s,source); check<std::uint32_t>(s,source);
            check<std::int64_t>(s,source); check<std::uint64_t>(s,source);
            check<float>(s,source); check<double>(s,source); check<char>(s,source); check<unsigned char>(s,source);
            check<std::vector<double>>(s,source); check<std::vector<std::string>>(s,source);
            check<signed char>(s,source); check<short>(s,source); check<unsigned short>(s,source);
            check<long long>(s,source); check<unsigned long long>(s,source); check<long double>(s,source);
            check<wchar_t>(s,source); check<char16_t>(s,source); check<char32_t>(s,source);
        }
    }
}

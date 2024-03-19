//
// Created by Ciaran on 30/04/2021.
//

#ifndef ROADRUNNER_SETTING_H
#define ROADRUNNER_SETTING_H

#include <variant>
#include <iostream>
#include <vector>
#include <sstream>
#include <limits>
#include <type_traits>
#include <functional>
#include <typeinfo>
#include "setting_t.h"
#include "rrException.h"

namespace rr {


    /**
     * @brief Store a roadrunner option (or setting) as a Variant type.
     * @details Uses a std::variant to hold one of the following types
     *      , std::monostate, std::string, bool, std::int32_t, std::uint32_t,
     *      std::int64_t, std::uint64_t, float, double,
     *      char, unsigned char, std::vector<double>
     * @note for developers: order of the std::variant template specialization is important.
     * Make sure the order always matches the other of the Setting::TypeId enum.
     * @note The function of this class used to be performed
     * by rr::Variant. Since C++17 we have replaced this with a
     * standard type.
     *
     * @code
     *  // implicit type conversions
     *  Setting setting(4); // is an int
     *  int s = setting.get<int>(); // okay
     *  int t = setting.get<std::string>(); // not okay, setting contains an int
     *  auto u = setting.get_if<int>(); // u is pointer to int if setting is int else nullptr
     *  if (auto v = setting.get_if<int>()){
     *      // do something in context where `setting` is int
     *  }
     *  Setting setting2(-4); // contains an int with negative value
     *  int q = setting2; // okay
     *  long r = setting2; // okay
     *  unsigned int w = setting2; // error, setting2 contains negative value
     *  unsigned long x = setting2; // error, setting2 contains negative value
     *
     *  long biggest_int = std::numeric_limits<int>::max();
     *  long out_of_range = biggest_int * 10;
     *  Setting setting(out_of_range);
     *  int x = setting; // error, int isn't big enough to hold `out_out_range`
     *
     *  // equality works as expected
     *  Setting setting3(std::int64_t(12345678912345));
     *  Setting setting4(std::int64_t(12345678912345));
     *  ASSERT_TRUE(setting3 == setting4); // pass
     *  ASSERT_TRUE(setting3 == 12345678912345); // also okay
     *
     *  // reassign a setting to different type
     *  Setting setting5(123);
     *  setting5 = "Now a string"; // okay
     *  std::string expected = "Now a string";
     *  ASSERT_TRUE(setting5 == expected); // pass
     *
     *  // instantiating a setting from other objects
     *  std::string st = "A String";
     *  Setting setting6(st); //okay
     *  Setting setting7 = st ; // not okay
     *  Setting setting8 = "A string"; // not okay
     *
     *  // check if Setting supports a type
     *  bool supported = Setting::isValidType<int>(); // true
     *  supported = Setting::isValidType<std::string>(); // true
     *  supported = Setting::isValidType<const char*>(); // false
     * @endcode
     */
    class Setting {
    public:

        /**
         * @brief constructor to take any type
         * accepted by setting_t and store as
         * a variant.
         */
        explicit Setting(setting_t value);

        /**
         * @brief constructor for creating a Setting
         * from a supperted type T.
         * @see setting_t for supported types
         * @details for developers: clang-tidy will want
         * you to make this constructor explicit. However,
         * we intentionally keep implicit to allow type casting
         * from type T to Setting.
         */
        template<class T>
        Setting(T settingValue) :
                value_(setting_t(settingValue)) {};

        /**
         * @brief constructor for enabling creation
         * of a Setting from a string literal. The literal
         * gets converted into a std::string
         * @code
         * Setting s("a string"); //interpreted as string, not const char*
         * @endcode
         */
        Setting(const char *settingValue);

        /**
         * @brief construct from a long.
         * (std::int64_t is a long long, which
         * matters on windows)
         */
        explicit Setting(std::int64_t  settingValue);


        /**
         * @brief default constructor. Allows instantiating
         * of an empty setting. The empty setting will
         * have a type of std::monostate, which is index
         * 0 in setting_t
         */
        Setting() = default;

        /**
         * @brief types that correspond to the index of the
         * position of the type in the variant template.
         * i.e. std::string is index 0.
         */
        enum TypeId {
            EMPTY = 0, // std::monostate. Empty is default when variant instantiated with nothing
            STRING = 1,
            BOOL = 2,
            INT32 = 3,
            UINT32 = 4,
            INT64 = 5,
            UINT64 = 6,
            FLOAT = 7,
            DOUBLE = 8,
            CHAR = 9,
            UCHAR = 10,
            DOUBLEVECTOR = 11,
        };

        /**
         * @brief returns the type of std::variant
         * contained within this Setting.
         * @returns an integer which refers to the index
         * of the type contained in this Setting. The index
         * is resolved by looking at the TypeId enum (or indeed
         * the ordering of the type parameters to setting_t.
         */
        TypeId type() const;

        /**
         * @brief determines whether the current value of
         * Setting is of type T.
         * @returns true is Setting is of type T
         */
        template<class T>
        bool isType() {
            return std::holds_alternative<T>(value_);
        }

        /**
         * @brief test for membership of type T in
         * setting_t, aka supported types
         */
        template<class T>
        static bool isValidType() {
            return isValidVariantType<T, setting_t>();
        }

        /**
         * @brief get the value of this Setting
         * as type T if the value in this Setting is
         * of type T.
         * @details wrapper around std::get_if
         */
        template<class T>
        T *get_if() {
            checkValidType<T>();
            return std::get_if<T>(&value_);
        }

        /**
         * @brief return the value held by this Setting
         * as a type SettingType.
         * @details simple wrapper around std::get.
         * @note for developers. This is a replacement for rr::Variant::convert in legacy
         * roadrunner. The name "get" is preferred to "convert" since it
         * follows C++ standard conventions.
         */
        template<class SettingType>
        SettingType get() {
            return std::get<SettingType>(value_);
        }

        /**
         * @brief explicitly convert this Setting to type As.
         * @details type contained by this Setting must be readily
         * convertible to type As and throws std::invalid_argument if not.
         * @code
         *  Setting setting(5);
         *  ASSERT_EQ(setting.get<int>(), 5); // fails if setting is not int
         *  ASSERT_THROW(setting.get<unsigned int>();, std::bad_variant_access); // bad, setting contains an int
         *  ASSERT_EQ(setting.getAs<unsigned int>(), 5); // Okay, we can convert from int to unsigned (when int > 0)
         * @endcode
         */
        template<class As>
        As getAs() const{
            const std::type_info &inf = typeInfo();
            return visit([&](auto &&val) {
                if constexpr (std::is_convertible_v<decltype(val), As>) {
                    std::ostringstream os;
                    os << "Cannot retrieve setting value:  you have requested the value as a ";
                    os << "\"" << typeid(As).name() << "\", but the value of the setting is ";
                    std::ostringstream oss_val;

                    // if we have some form of negative signed integer
                    bool isNegInt = false;
                    uint64_t uint64val = 0;
                    int64_t int64val = 0;
                    if (auto intValue = std::get_if<int>(&value_))
                    {
                        if (*intValue < 0)
                        {
                            isNegInt = true;
                            int64val = *intValue;
                        }
                        uint64val = *intValue;
                        oss_val << "\"" << *intValue << "\", which is ";
                    }
                    else if (auto intValue = std::get_if<int64_t>(&value_))
                    {
                        if (*intValue < 0)
                        {
                            isNegInt = true;
                            int64val = *intValue;
                        }
                        oss_val << "\"" << *intValue << "\", which is ";
                        uint64val = *intValue;
                    }
                    else if (auto intValue = std::get_if<unsigned int>(&value_))
                    {
                        oss_val << "\"" << *intValue << "\", which is ";
                        uint64val = *intValue;
                    }
                    else if (auto intValue = std::get_if<uint64_t>(&value_))
                    {
                        oss_val << "\"" << *intValue << "\", which is ";
                        uint64val = *intValue;
                    }
                    if (isNegInt)
                    {
                        // we cannot convert to an unsigned type
                        if (typeid(As) == typeid(unsigned int) || typeid(As) == typeid(unsigned long)) {
                            // would prefer to throw std::bad_variant_access, but it seems
                            // it does not have the appropriate constructor (?)
                            os << oss_val.str() << "negative." << std::endl;
                            throw std::invalid_argument(os.str());
                            return As{}; // oddly enough, this *is* necessary
                        }
                    }
                    // if we have a double, with value greater than std::numeric_limits<float>::max
                    // and we try to convert to float, we have an error
                    if (auto lValue = std::get_if<float>(&value_)) {
                        if (*lValue > ((std::numeric_limits<float>::max)())) {
                            os << "\"" << *lValue << "\", which is too large." << std::endl;
                            throw std::invalid_argument(os.str());
                            return As{}; // must be present
                        }
                    }
                    //Now check if the value is larger than the max value for the type:
                    if (typeid(As) == typeid(int) &&
                        (uint64val > ((std::int64_t)(std::numeric_limits<int>::max)()) && !isNegInt) ||
                            (isNegInt && int64val < ((std::int64_t)(std::numeric_limits<int>::min)())))
                    {
                        os << oss_val.str() << "too large." << std::endl;
                        throw std::invalid_argument(os.str());
                        return As{}; // must be present
                    }
                    else if (typeid(As) == typeid(unsigned int) &&
                        uint64val > ((std::int64_t)(std::numeric_limits<unsigned int>::max)()))
                    {
                        os << oss_val.str() << "too large." << std::endl;
                        throw std::invalid_argument(os.str());
                        return As{}; // must be present
                    }
                    else if (typeid(As) == typeid(int64_t) &&
                        (uint64val > ((std::int64_t)(std::numeric_limits<int64_t>::max)()) && !isNegInt))
                    {
                        os << oss_val.str() << "too large." << std::endl;
                        throw std::invalid_argument(os.str());
                        return As{}; // must be present
                    }
                    return As(val);
                } else {
                    std::ostringstream os;
                    os << "Setting::getAs:TypeError. You have requested the conversion "
                          "of a \"" << typeid(decltype(val)).name() << "\" to a ";
                    os << "\"" << typeid(As).name() << "\" but this Setting contains ";
                    os << "a \"" << inf.name() << "\". Note, see Setting::toString() "
                                                  "for string representation." << std::endl;
                    // would prefer to throw std::bad_variant_access, but it seems
                    // it does not have the appropriate constructor (?)
                    throw std::invalid_argument(os.str());
                    return As{}; // oddly enough, this *is* necessary
                }
            });
        }

        /**
         * @brief return the value held by this setting as type SettingType.
         * @details const version of SettingType::get()
         * @see SettingType::get()
         */
        template<class SettingType>
        SettingType get() const {
            return std::get<SettingType>(value_);
        }

        /**
         * @brief implicit type conversion support for the Setting class.
         * @details this method is responsible for enabling implicit conversion
         * between Setting and its supported types. Moreover, it enables
         * one to implicitly convert (say) a int to a long, in contexts
         * that make sense (as determined by std::is_convertible_v).
         * Some situations are explicitly prohibited from implicit type
         * conversion. These include
         *  - converting a negative int or long to unsigned int or long
         *  - converting a long that has a value > the largest possible value for int to int
         * @note for developers: clang-tidy will ask you to make this
         * explicit, but this method *must* not be explicit since its
         * entire purpose is implicit type conversion.
         * @note for developers: Any additions to types stored in a setting_t
         * *must* have appropriate unit tests added for the new types in
         * SettingTests.cpp.
         */
        template<typename T>
        operator T() const {
            return std::visit(
                    [&](auto const &val) {
                        if constexpr (std::is_convertible_v<decltype(val), T>) {
                            // if all is good, we construct a object of type T and return it.
                            return getAs<T>();
                        } else {
                            // T is not supported conversion type from typeid(value_)
                            throw std::bad_variant_access{};
                            return T{}; // oddly enough, this *is* necessary
                        }
                    }, value_);
        }

        /**
         * @brief implicit cast this Setting to a
         * std::vector<double>. If this operator is
         * used and the type contained by this Setting
         * is not a double vector, an std::invalid_argument
         * error is thrown.
         * @details the generic operator T() method does not
         * cover the case when we have a vector of doubles, so
         * we must implement this manually. We could write another
         * template for the more general case, but only support for
         * std::vector<double> is needed.
         */
        operator std::vector<double>() {
            if (auto vec = get_if<std::vector<double>>()) {
                return *vec;
            } else {
                throw std::invalid_argument("Setting::operator std::vector<double>: "
                                            "could not cast Setting to double vector because the "
                                            "type contained in this Setting is not a double vector");
            }
        }


        /**
         * @brief equality operator for comparing object
         * otherSetting of type T against the Setting
         * contained within this Setting.value_.
         * Enables the following types of comparisons
         * @code
         *  Setting s1("string");
         *  std::string s2 = "string";
         *  s1 == s2; // true
         *  // or
         *  Setting s3(1234);
         *  int s4 = 1234;
         *  s3 == s4; // true
         * @details Implemented using SFINAE - substitute failure is not an error.
         * The seconds template argument is only true when type T is a valid variant
         * and when this argument evaluates to false, the compiler will not generate
         * a code for the template types. Thus, this template is only defined for
         * types that are a part of setting_t.
         */
        template<typename T, class = typename std::enable_if<isValidVariantType<T, setting_t>::value>::type>
        bool operator==(const T &otherSetting) {
            if (auto settingValue = get_if<T>()) {
                return *settingValue == otherSetting;
            }
            return false;
        }

        /**
         * @brief Inequality operator, counter part to equality operator
         * @see operator==(const T&otherSetting);
         */
        template<typename T,
                class = typename std::enable_if<isValidVariantType<T, setting_t>::value>::type>
        bool operator!=(const T &setting) {
            return !(*this == setting);
        }

        /**
         * @brief Equality operator, enabling the comparison
         * of this Setting with the other Setting
         */
        bool operator==(const Setting &setting);

        /**
         * @brief Equality operator, enabling the comparison
         * of this Setting with string literals
         * @details since our Setting does not store
         * const char*, the main templated `operated==(const T&)`
         * is not used. We therefore define this separately
         */
        bool operator==(const char *setting);

        /**
         * @brief Equality operator, enabling the comparison
         * of this Setting with string literals
         * @details since our Setting does not store
         * const char*, the main templated `operated==(const T&)`
         * is not used. We therefore define this separately.
         */
        bool operator!=(const char *setting);

        /**
         * @brief Inequality operator, enabling the comparison
         * of this Setting with the other Setting
         */
        bool operator!=(const Setting &setting);


        /**
         * @brief assignment operator for objects that are not
         * a setting. T must be a member of setting_t or this
         * template will fail without error (SFINAE)
         */
        template<typename T, class = typename std::enable_if<isValidVariantType<T, setting_t>::value>::type>
        Setting &operator=(const T &setting) {
            checkValidType<T>();
            // no need to check self assignment with variant
            value_ = setting_t(setting);
            return *this;
        }

        /**
         * @brief move assignment operator
         * for types T that are members of setting_t.
         * @see setting_t
         */
        template<typename T, class = typename std::enable_if<isValidVariantType<T, setting_t>::value>::type>
        Setting &operator=(T &&setting) noexcept {
            checkValidType<T>();
            // no need to check self assignment with variant
            value_ = std::move(setting_t(setting));
            return *this;
        }

        /**
         * @brief wrapper around std::visitor for
         * "visiting" each of the possible types in
         * the setting_t.
         * @details instead of using an if-else block
         * you can use a visitor. For instance,
         * @param function: callable type, such as lambda expression.
         * @code
         *     Setting s(1234);
         *     std::string typeString;
         *     s.visit([&](auto t) -> decltype(auto){typeString = typeid(t).name();});
         *     std::cout << typeString << std::endl; // prints out "int" to console
         * @endcode
         * @see docs on std::visit
         */
        template<class Func>
        decltype(auto) visit(Func function) const {
            return std::visit(function, value_);
        }

        /**
         * @brief returns the std::type_info for the
         * object type contained within this Setting.
         * (aka one of the types within setting_t)
         */
        const std::type_info &typeInfo() const;


        /**
         * @brief move assignment operator for when
         * rhs is a Setting.
         * @see operator=(const T&setting);
         */
        Setting &operator=(const Setting &setting);

        /**
         * @brief Move assignment operator for Setting
         * with another Setting
         */
        Setting &operator=(Setting &&setting) noexcept;

        /**
         * @brief Copy constructor for Setting
         * with another Setting
         */
        Setting(const Setting &setting);

        /**
         * @brief Move constructor for Setting
         * with another Setting
         */
        Setting(Setting &&setting) noexcept;

        /**
         * @brief getter for the std::variant_t underlying this
         * Setting.
         */
        [[nodiscard]] const setting_t &getValue() const;

        /**
         * @author JKM
         * @brief Convert to Python-compatible representation
         */
        [[nodiscard]] std::string pythonRepr() const;

        /**
         * @brief is this variant a std::string.
         */
        [[nodiscard]] bool isString() const;

        /**
         * @brief was an integer stored here.
         */
        [[nodiscard]] bool isInteger() const;

        /**
         * @brief is this a numeric type.
         */
        [[nodiscard]] bool isNumeric() const;

        /**
         * @brief is this a boolean type.
         */
        [[nodiscard]] bool isBool() const;

        /**
         * @brief true if empty.
         */
        bool isEmpty() const;

        /**
         * @brief true if this is a signed number.
         */
        bool isSigned() const;

        /**
        * true if this is a std::vector of doubles
        */
        bool isDoubleVector() const;

        /**
        * true if this is a double
        */
        bool isDouble() const;

        /**
         * Parses the std::string which must be in JSON format. This is a common
         * way to read a Variant from a file or create a new one from a std::string:
         * @code
         * Variant v = Variant::parse("0.123");
         * @endcode
         * @note this function is a tangent to the main Setting class.
         * We should instead have a JSON parser, is that's the function
         * we need.
         */
        static Setting parse(std::string &val);

        /**
         * @brief gets a string representation
         * of the type stored in this Setting.
         */
        std::string toString();

    private:

        setting_t value_;


        /**
         * @brief throws std::invalid_argument if
         * type T is not supported by Setting.
         * @see setting_t
         */
        template<class T>
        void checkValidType() {
            if (!Setting::isValidType<T>()) {
                std::ostringstream err;
                err << "Setting does not support ";
                err << "type \"" << typeid(T).name() << "\"";
                throw std::invalid_argument(err.str());
            }
        }
    };

}
#endif //ROADRUNNER_SETTING_H

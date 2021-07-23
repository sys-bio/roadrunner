//
// Created by Ciaran on 05/05/2021.
//

#ifndef ROADRUNNER_SETTING_T_H
#define ROADRUNNER_SETTING_T_H

namespace rr{
    /**
     * Note: This code was isolated form the Setting.h because
     * swig had difficulty reading the isValidVariant template.
     */

    /**
     * @brief A setting type, used in the Setting class
     * to store multiple types.
     */
    using setting_t = std::variant<
            std::monostate, /*void, see docs for std::variant*/
            std::string, bool, std::int32_t,
            std::uint32_t, std::int64_t, std::uint64_t,
            float, double,
            char, unsigned char,
            std::vector<double>,
            std::vector<std::string>
            >;


    /**
     * @brief Generic type checking mechanism
     * for membership of type T in variant ALL_T.
     * @code
     *  isValidVariantType<int, setting_t > truth;
     *  ASSERT_TRUE(truth); // pass
     * @endcode
     */
    template<typename T, typename ALL_T>
    struct isValidVariantType;

    template<typename T, typename... ALL_T>
    struct isValidVariantType<T, std::variant<ALL_T...>>
            : public std::disjunction<std::is_same<T, ALL_T>...> {
    };

}

#endif //ROADRUNNER_SETTING_T_H

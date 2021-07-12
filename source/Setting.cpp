//
// Created by Ciaran on 30/04/2021.
//

#include "Setting.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "rrStringUtils.h"

namespace rr {

    Setting::Setting(const char *settingValue)
            : value_(std::string(settingValue)) {}

    Setting::Setting(std::int64_t settingValue)
            : value_(std::int64_t (settingValue)) {}

    Setting::Setting(setting_t value)
            : value_(std::move(setting_t(std::move(value)))) {};

    Setting::TypeId Setting::type() const {
        return (Setting::TypeId) value_.index();
    }

    bool Setting::isString() const {
        return type() == STRING;
    }

    bool Setting::isInteger() const {
        return type() == INT32 || type() == UINT32 || type() == INT64 || type() == UINT64;
    }

    bool Setting::isNumeric() const {
        return isInteger() || type() == FLOAT || type() == DOUBLE;
    }

    bool Setting::isBool() const {
        return type() == BOOL;
    }

    bool Setting::isEmpty() const {
        return type() == EMPTY;
    }

    bool Setting::isSigned() const {
        return type() == INT32 || type() == INT64 || type() == UCHAR;
    }

    bool Setting::isDoubleVector() const {
        return type() == DOUBLEVECTOR;
    }

    bool Setting::isDouble() const {
        return type() == DOUBLE;
    }

    std::string Setting::pythonRepr() const {
        if (isBool()) {
            if (get<bool>()) {
                return "True";
            } else {
                return "False";
            }
        } else if (isString()) {
            return "'" + get<std::string>() + "'";
        }
        return get<std::string>();
    }

    const setting_t &Setting::getValue() const {
        return value_;
    }

    Setting Setting::parse(std::string &s) {

        while (!s.empty() && std::isspace(*s.begin()))
            s.erase(s.begin());

        while (!s.empty() && std::isspace(*s.rbegin()))
            s.erase(s.length() - 1);

        const char *input = s.c_str();
        char *end = nullptr;

        // check for int
        int i = (int) strtol(input, &end, 0);
        if (*input != '\0' && end != input && *end == '\0') {
            return Setting(i);
        }

        // check for double
        double d = strtod(input, &end);
        if (*input != '\0' && end != input && *end == '\0') {
            return Setting(d);
        }

        // check for bool
        std::string bstr = s;
        std::transform(bstr.begin(), bstr.end(), bstr.begin(), ::toupper);

        if (bstr == "TRUE") {
            return Setting(true);
        }

        if (bstr == "FALSE") {
            return Setting(false);
        }

        //Check if std::vector of doubles
        if (s[0] == '[') {
            return Setting(toDoubleVector(s));
        }

        // its a std::string
        return Setting(s);
    }

    Setting &Setting::operator=(const Setting &setting) {
        if (this != &setting) {
            value_ = setting.getValue();
        };
        return *this;
    }

    Setting &Setting::operator=(Setting &&setting) noexcept {
        if (this != &setting) {
            value_ = setting.getValue();
        };
        return *this;
    }

    Setting::Setting(const Setting &setting)
            : value_(setting.getValue()) {}

    Setting::Setting(Setting &&setting) noexcept
            : value_(std::move(setting.getValue())) {}

    const std::type_info &Setting::typeInfo() const {
        return std::visit([](auto &&x) -> decltype(auto) {
            return std::move(typeid(x));
        }, value_);
    }

    bool Setting::operator==(const Setting &setting) {
        return (value_ == setting.value_);
    }

    bool Setting::operator==(const char *setting) {
        return (*this == std::string(setting));
    }

    bool Setting::operator!=(const char *setting) {
        return !(*this == setting);
    }

    bool Setting::operator!=(const Setting &setting) {
        return !(value_ == setting.value_);
    }

    std::string Setting::toString() {
        std::ostringstream os;
        os << std::boolalpha;
        if (auto v = get_if<std::monostate>()) {
            os << "None";
        } else if (auto v = get_if<std::vector<double>>()) {
            os << "[";
            for (int i = 0; i < v->size(); i++) {
                if (i == v->size() - 1) {
                    os << (*v)[i] << "]";
                } else {
                    os << (*v)[i] << ", ";
                }
            }
        } else if (auto v = get_if<std::string>()) {
            os << "'" << *v << "'";
        } else if (auto v = get_if<bool>()) {
            os << *v;
        } else if (auto v = get_if<std::int32_t>()) {
            os << *v;
        } else if (auto v = get_if<std::uint32_t>()) {
            os << *v;
        } else if (auto v = get_if<std::int64_t>()) {
            os << *v;
        } else if (auto v = get_if<std::int64_t>()) {
            os << *v;
        } else if (auto v = get_if<float>()) {
            os << *v;
        } else if (auto v = get_if<bool>()) {
            os << *v;
        } else if (auto v = get_if<double>()) {
            os << *v;
        }
        return os.str();
    }



}

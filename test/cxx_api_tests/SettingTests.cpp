//
// Created by Ciaran on 30/04/2021.
//

#include "gtest/gtest.h"
#include "Setting.h"
#include <exception>
#include <unordered_map>

using namespace rr;

class SettingTests : public ::testing::Test {
public:
    SettingTests() = default;

};

/******************************************************
 * Tests for storage of types used in Roadrunner
 */


TEST_F(SettingTests, TestSettingTypeMonostate) {
    Setting setting;
    ASSERT_EQ(setting.type(), Setting::TypeId::EMPTY);
}

TEST_F(SettingTests, TestSettingTypeString) {
    Setting setting("Really high");
    ASSERT_EQ(setting.type(), Setting::TypeId::STRING);

}

TEST_F(SettingTests, TestSettingTypeBool) {
    Setting setting(true);
    ASSERT_EQ(setting.type(), Setting::TypeId::BOOL);
}

TEST_F(SettingTests, TestSettingTypeInt32_t) {
    Setting setting(std::int32_t(-354));
    ASSERT_EQ(setting.type(), Setting::TypeId::INT32);
}

TEST_F(SettingTests, TestSettingTypeUint32_t) {
    Setting setting(std::uint32_t(1234));
    ASSERT_EQ(setting.type(), Setting::TypeId::UINT32);
}

TEST_F(SettingTests, TestSettingTypeInt64_t) {
    Setting setting(std::int64_t(-4512365));
    ASSERT_EQ(setting.type(), Setting::TypeId::INT64);
}

TEST_F(SettingTests, TestSettingTypeUint64_t) {
    Setting setting(std::uint64_t(24565434));
    ASSERT_EQ(setting.type(), Setting::TypeId::UINT64);
}

TEST_F(SettingTests, TestSettingTypeFloat) {
    Setting setting(float(0.12346));
    ASSERT_EQ(setting.type(), Setting::TypeId::FLOAT);
}

TEST_F(SettingTests, TestSettingTypeDouble) {
    Setting setting(double(0.1324688));
    ASSERT_EQ(setting.type(), Setting::TypeId::DOUBLE);
}

TEST_F(SettingTests, TestSettingTypeChar) {
    Setting setting('c');
    ASSERT_EQ(setting.type(), Setting::TypeId::CHAR);
}

TEST_F(SettingTests, TestSettingTypeUnsignedChar) {
    unsigned char x = 'x';
    Setting setting(x);
    ASSERT_EQ(setting.type(), Setting::TypeId::UCHAR);
}

TEST_F(SettingTests, TestSettingTypeDoubleVector) {
    Setting setting(std::vector<double>({0.1, 0.2, 0.3}));
    ASSERT_EQ(setting.type(), Setting::TypeId::DOUBLEVECTOR);
}


/******************************************************
 * Implicit cast tests, but the simple case.
 * Extracting an int from the setting for instance,
 * and no int to long.
 */

TEST_F(SettingTests, TestImplicitCastToMonostate) {
    Setting setting;
    std::monostate x = setting;
    ASSERT_EQ(typeid(x), typeid(std::monostate));
}

TEST_F(SettingTests, TestImplicitCastToString) {
    Setting setting("Really high");
    std::string x = setting;
    ASSERT_EQ(typeid(x), typeid(std::string));
}

TEST_F(SettingTests, TestImplicitCastToStringAndDoSomethingWithIt) {
    Setting setting("Really high");
    std::string x = setting;
    bool truth = x.find("Really", 0) != std::string::npos;
    ASSERT_TRUE(truth);
}


TEST_F(SettingTests, TestImplicitCastToBool) {
    Setting setting(true);
    bool x = setting;
    ASSERT_EQ(typeid(x), typeid(bool));
}

TEST_F(SettingTests, TestImplicitCastToInt32_t) {
    Setting setting(std::int32_t(-354));
    std::int32_t x = setting;
    ASSERT_EQ(typeid(x), typeid(std::int32_t));
}

TEST_F(SettingTests, TestImplicitCastToUint32_t) {
    Setting setting(std::uint32_t(1234));
    std::uint32_t x = setting;
    ASSERT_EQ(typeid(x), typeid(std::uint32_t));
}

TEST_F(SettingTests, TestImplicitCastToInt64_t) {
    Setting setting(std::int64_t(-451236));
    std::int64_t x = setting;
    ASSERT_EQ(typeid(x), typeid(std::int64_t));
}

TEST_F(SettingTests, TestImplicitCastToUint64_t) {
    Setting setting(std::uint64_t(2456543));
    std::uint64_t x = setting;
    ASSERT_EQ(typeid(x), typeid(std::uint64_t));
}

TEST_F(SettingTests, TestImplicitCastToFloat) {
    Setting setting(float(0.12346));
    float x = setting;
    ASSERT_EQ(typeid(x), typeid(float));
}

TEST_F(SettingTests, TestImplicitCastToDouble) {
    Setting setting(double(0.1324688));
    double x = setting;
    ASSERT_EQ(typeid(x), typeid(double));
}

TEST_F(SettingTests, TestImplicitCastToChar) {
    Setting setting('c');
    char x = setting;
    ASSERT_EQ(typeid(x), typeid(char));
}

TEST_F(SettingTests, TestImplicitCastToUnsignedChar) {
    unsigned char y = 'y';
    Setting setting(y);
    unsigned char x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned char));
}

TEST_F(SettingTests, TestImplicitCastToDoubleVector) {
    Setting setting(std::vector<double>({0.123, 0.456, 0.789}));
    std::vector<double> x = setting;
    ASSERT_EQ(typeid(x), typeid(std::vector<double>));
}

/*********************************************************
 * Conversions between relevant different types.
 * We do not test all combinations of type pairs,
 * only the ones which are supposed to work.
 * All other pairs, i.e. incompatible pairs like int to string
 * are not tested and should throw std::bad_variant_access.
 */

TEST_F(SettingTests, ImplicitCastIntToUInt) {
    Setting setting(4);
    unsigned int x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned int));
}

TEST_F(SettingTests, ImplicitCastIntToLong) {
    Setting setting(4);
    long x = setting;
    ASSERT_EQ(typeid(x), typeid(long));
}

TEST_F(SettingTests, ImplicitCastIntToULong) {
    Setting setting(4);
    unsigned long x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned long));
}

TEST_F(SettingTests, ImplicitCastNegativeIntToUInt) {
    // should raise error
    Setting setting(-4);
    ASSERT_THROW(
            unsigned int x = setting,
            std::invalid_argument
    );
}

TEST_F(SettingTests, ImplicitCastNegativeIntToULong) {
    // should raise error
    Setting setting(-4);
    ASSERT_THROW(
            unsigned long x = setting,
            std::invalid_argument
    );
}

TEST_F(SettingTests, ImplicitCastLongToIntOutOfBounds) {
    // should raise error
    std::int64_t biggest_int = std::numeric_limits<int>::max();
    std::int64_t out_of_range = biggest_int * 10;
    Setting setting(out_of_range);
    ASSERT_THROW(
            int x = setting,
            std::invalid_argument
    );
}

TEST_F(SettingTests, ImplicitCastLongToNegIntOutOfBounds) {
    // should raise error
    std::int64_t biggest_int = std::numeric_limits<int>::max();
    std::int64_t out_of_range = -biggest_int * 10;
    Setting setting(out_of_range);
    ASSERT_THROW(
        int x = setting,
        std::invalid_argument
    );
}

TEST_F(SettingTests, ImplicitCastLongToInt) {
    Setting setting(std::int64_t(4));
    int x = setting;
    ASSERT_EQ(typeid(x), typeid(int));
}

TEST_F(SettingTests, ImplicitCastLongToUInt) {
    Setting setting(std::int64_t(4));
    unsigned int x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned int));
}

TEST_F(SettingTests, ImplicitCastLongToULong) {
    Setting setting(std::int64_t(4));
    unsigned long x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned long));
}

TEST_F(SettingTests, ImplicitCastStringToConstCharStar) {
    Setting setting(std::int64_t(4));
    unsigned long x = setting;
    ASSERT_EQ(typeid(x), typeid(unsigned long));
}

TEST_F(SettingTests, ImplicitInstantiateFromBool) {
    Setting setting = true;
    std::ostringstream ss;
    ss << std::boolalpha << *setting.get_if<bool>() << std::endl;
    ASSERT_STREQ(ss.str().c_str(), "true\n");
}

TEST_F(SettingTests, ImplicitInstantiateFromBoolInAMap) {
    std::unordered_map<std::string, Setting> m({
                                                       {"isABool", true}
                                               });
    Setting setting = m["isABool"];
    ASSERT_TRUE(*setting.get_if<bool>());
}

TEST_F(SettingTests, ImplicitInstantiateFromStringInAMap) {
    std::unordered_map<std::string, Setting> m({
                                                       {"isABool", "AString"}
                                               });
    Setting setting = m["isABool"];
    ASSERT_STREQ("AString", (*setting.get_if<std::string>()).c_str());
}

TEST_F(SettingTests, GetUnsignedIntFromInt) {
    Setting setting(5);
    ASSERT_EQ(setting.get<int>(), 5); // fails if setting is not int
    ASSERT_THROW(setting.get<unsigned int>();, std::bad_variant_access); // bad, setting contains an int
    ASSERT_EQ(setting.getAs<unsigned int>(), 5); // Okay, we can convert from int to unsigned when int > 0
}

TEST_F(SettingTests, GetAsBadTypeRequest) {
    Setting setting(5);
    // setting contains an int
    ASSERT_EQ(setting.get<int>(), 5);
    // but we request conversion from unsigned int
    ASSERT_THROW(
    setting.getAs<std::string>();,
    std::invalid_argument
    );
}

/********************************************
 * Getting values from Setting
 */

TEST_F(SettingTests, GetIfTestWithString) {
    Setting setting("I'm a string");
    auto string = setting.get_if<std::string>();
    ASSERT_STREQ(string->c_str(), "I'm a string");
}

TEST_F(SettingTests, GetIfTestWithInt) {
    Setting setting(1234);
    auto val = setting.get_if<int>();
    ASSERT_EQ(*val, 1234);
}

TEST_F(SettingTests, GetIfWhenValueNotRightType) {
    Setting setting(1234);
    auto val = setting.get_if<std::int64_t>();
    ASSERT_EQ(val, nullptr);
}

TEST_F(SettingTests, PutSettingInAVectorAndEqualityOp) {
    Setting setting("I'm a string");
    std::vector<Setting> v({setting});
    Setting x = v[0];
    bool truth = x == setting;
    ASSERT_TRUE(truth);
}

TEST_F(SettingTests, ImplicitCastFromSettingInAVector) {
    Setting setting("I'm a string");
    std::vector<Setting> v({setting});
    std::string x = v[0];
    ASSERT_EQ(x, "I'm a string");
}


TEST_F(SettingTests, EqualityOperatorWithStringsForwards) {
    Setting setting1("I'm a setting");
    Setting setting2("I'm a setting");
    ASSERT_TRUE(setting1 == setting2);
}

TEST_F(SettingTests, EqualityOperatorWithStringsBackwards) {
    Setting setting1("I'm a setting");
    Setting setting2("I'm a setting");
    ASSERT_TRUE(setting2 == setting1);
}

TEST_F(SettingTests, InEqualityOperatorWithStringsForwards) {
    Setting setting1("I'm a setting");
    Setting setting2("I'm another setting");
    ASSERT_TRUE(setting1 != setting2);
}

TEST_F(SettingTests, InEqualityOperatorWithStringsBackwards) {
    Setting setting1("I'm a setting");
    Setting setting2("I'm another setting");
    ASSERT_TRUE(setting2 != setting1);
}

/**
 * This test fails on mac since the equality operator
 * is ambigious. Since this particular comparison
 * is not used in the main code its a low priority.
 * If ever in future we need this comparison,
 * we can use §setting.getAs<std::int64_t>()
 */
TEST_F(SettingTests, DISABLED_EqualityWithTemporaryLongFwd) {
    Setting setting(std::int64_t(12345678912345));
//    ASSERT_TRUE(setting == 12345678912345);
}

TEST_F(SettingTests, DISABLED_EqualityWithTemporaryLongBkwd) {
    Setting setting(std::int64_t(12345678912345));
//    ASSERT_TRUE(12345678912345 == setting); // compile error. Limitation of strategy. Could use https://stackoverflow.com/questions/67374983/how-to-use-overload-operator-in-a-stdvariant-wrapper-class-to-make-compa/67375487#67375487
}

TEST_F(SettingTests, EqualityOperatorWithIntFwd) {
    Setting setting(345);
    int x = 345;
    ASSERT_TRUE(setting == x);
}

TEST_F(SettingTests, DISABLED_EqualityOperatorWithIntBkwd) {
    Setting setting(345);
    int x = 345;
//    ASSERT_TRUE(x == setting ); // fail. Limitation with Stratgy.
}

TEST_F(SettingTests, InEqualityOperatorWithIntFwd) {
    Setting setting(345);
    int x = 3456;
    ASSERT_FALSE(setting == x);
}

TEST_F(SettingTests, DISABLED_InEqualityOperatorWithIntBkwd) {
    Setting setting(345);
    int x = 3456;
//    ASSERT_FALSE(x == setting ); // fails. Limitation of Setting strategy. Could use https://stackoverflow.com/questions/67374983/how-to-use-overload-operator-in-a-stdvariant-wrapper-class-to-make-compa/67375487#67375487
}

TEST_F(SettingTests, CheckIsTypeWhenTrue) {
    Setting setting(345);
    bool x = setting.isType<int>();
    ASSERT_TRUE(x);
}

TEST_F(SettingTests, CheckIsTypeWhenFalse) {
    Setting setting(345);
    bool x = setting.isType<std::vector<double>>();
    ASSERT_FALSE(x);
}

TEST_F(SettingTests, CheckTypeIntIsValid) {
    bool truth = Setting::isValidType<int>();
    ASSERT_TRUE(truth);
}

TEST_F(SettingTests, CheckTypeConstCharStarIsInvalid) {
    bool truth = Setting::isValidType<const char *>();
    ASSERT_FALSE(truth);
}

TEST_F(SettingTests, AssignmentOperatorSettingToSetting) {
    Setting setting(345);
    setting = Setting("string");
    std::string expected = "string";
    ASSERT_TRUE(setting == expected);
}

TEST_F(SettingTests, AssignmentOperatorIntToString) {
    Setting setting(345);
    setting = std::string("string");
    std::string expected = "string";
    ASSERT_TRUE(setting == expected);
}

TEST_F(SettingTests, AssignmentOperatorMonostateToBool) {
    Setting setting;
    setting = false;
    ASSERT_FALSE(setting);
}

TEST_F(SettingTests, AssignmentOperatorUIntToDouble) {
    Setting setting(123456u);
    setting = 0.12345;
    ASSERT_EQ(setting.get<double>(), 0.12345);
}

void functionThatTakesASetting(bool truth, Setting &setting) {
    setting = truth;
}

TEST_F(SettingTests, PassingASettingToFunction) {
    Setting setting;
    functionThatTakesASetting(true, setting);
    ASSERT_TRUE(setting == true);
}

TEST_F(SettingTests, InAMap) {
    std::unordered_map<std::string, Setting> smap;
    smap["A String"] = Setting("String");
    ASSERT_STREQ(smap["A String"].get<std::string>().c_str(), "String");
}


TEST_F(SettingTests, InAMapAssignNewValue) {
    std::unordered_map<std::string, Setting> smap;
    smap["A String"] = Setting("String");
    smap["A String"] = true;
    if (auto isBool = smap["A String"].get_if<bool>()) { // nullptr if not a bool
        ASSERT_TRUE(typeid(*isBool) == typeid(bool));
        return;
    }
    ASSERT_TRUE(false);//if we hit here we've failed
}

TEST_F(SettingTests, InAMapImplicitInstantiationOfInt) {
    std::unordered_map<std::string, Setting> smap;
    smap["AnInt"] = 10;
    ASSERT_EQ(typeid(smap["AnInt"].getAs<int>()), typeid(int));
}


struct TypeThatContainsASettingsMap {
    TypeThatContainsASettingsMap() {
        settingsMap["string"] = Setting("string");
        settingsMap["int"] = Setting(1);
        settingsMap["long"] = Setting(std::int64_t(100));
    };

    void setValue(const std::string &key, Setting value) {
        settingsMap[key] = std::move(value);
    }

    std::unordered_map<std::string, Setting> settingsMap{};
};

/**
 * This test helped debug an issue whereby
 * Setting cannot be implicitly instantiated
 */
TEST_F(SettingTests, TypeThatUsesSettingsMapChangeTheInt) {
    TypeThatContainsASettingsMap t;
    t.setValue("int", 15);
    ASSERT_TRUE(t.settingsMap["int"] == 15);
}

TEST_F(SettingTests, TypeThatUsesSettingsMapChangeTheString) {
    TypeThatContainsASettingsMap t;
    t.setValue("string", std::string("ANewString"));
    ASSERT_TRUE(t.settingsMap["string"] == "ANewString");
}

TEST_F(SettingTests, TypeThatUsesSettingsMapChangeTheStringFromLiteral) {
    TypeThatContainsASettingsMap t;
    t.setValue("string", "ANewString");
    ASSERT_TRUE(t.settingsMap["string"] == "ANewString");
}

TEST_F(SettingTests, EqualityWithString) {
    Setting setting("ANewString");
    ASSERT_TRUE(setting == "ANewString");
}

TEST_F(SettingTests, CheckThatWhenYouCreateAStringSettingThatItIsAStringNotBool) {
    Setting setting("ANewString");
    if (auto strVal = setting.get_if<std::string>()) {
        std::cout << "setting is a string" << std::endl;
        ASSERT_STREQ((*strVal).c_str(), "ANewString");
    } else {
        ASSERT_TRUE(false && "We've failed the test, input string is not a string");
    }
}

TEST_F(SettingTests, TypeThatUsesSettingsMapChangeUsingASetting) {
    TypeThatContainsASettingsMap t;
    t.setValue("string", Setting(std::string()));
    ASSERT_TRUE(t.settingsMap["string"] == std::string());
}

/**
 * Keep the test in case we figure out a way to implement this
 */
TEST_F(SettingTests, DISABLED_AutomaticTypeDeductionInGet) {
    Setting setting(1234l);
    // error this is not supported right now. Probably possible to do
    // with the likes of decltype(auto).
    // long x = setting.get(); // error,
    // ASSERT(x == 1234l);
}

TEST_F(SettingTests, CastFromSettingToDoubleVector) {
    Setting setting(std::vector<double>({0.1, 0.2}));

}


TEST_F(SettingTests, TypeInfoIsAccurateForInt) {
    Setting s(1234);
    ASSERT_EQ(s.typeInfo(), typeid(int));
}


TEST_F(SettingTests, UseVisitorWrapper) {
    Setting s(1234);
    ASSERT_EQ(typeid(int), s.visit([&](auto t) -> decltype(auto) { return typeid(t); }));
}


















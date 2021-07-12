//
// Created by Ciaran on 20/05/2021.
//

#include "gtest/gtest.h"
#include "Dictionary.h"

using namespace rr;

class BasicDictionaryTests : public ::testing::Test {
public:
    BasicDictionaryTests() = default;
};

TEST_F(BasicDictionaryTests, setItem){
    BasicDictionary basicDictionary;
    basicDictionary.setItem("AnInt", 4);
    ASSERT_EQ((int)basicDictionary.getItem("AnInt"), 4);
}

TEST_F(BasicDictionaryTests, hasKeyWhenTrue){
    BasicDictionary basicDictionary;
    basicDictionary.setItem("AString", "StringyString");
    ASSERT_TRUE(basicDictionary.hasKey("AString"));
}
TEST_F(BasicDictionaryTests, hasKeyWhenFalse){
    BasicDictionary basicDictionary;
    basicDictionary.setItem("AString", "StringyString");
    ASSERT_FALSE(basicDictionary.hasKey("AStringyString"));
}

TEST_F(BasicDictionaryTests, deleteItem){
    BasicDictionary basicDictionary;
    basicDictionary.setItem("AString", "StringyString");
    ASSERT_TRUE(basicDictionary.hasKey("AString"));
    basicDictionary.deleteItem("AString");
    ASSERT_FALSE(basicDictionary.hasKey("AString"));
}

TEST_F(BasicDictionaryTests, getKeys){
    BasicDictionary basicDictionary;
    basicDictionary.setItem("AString", "StringyString");
    std::vector<std::string> expected({"AString"});
    ASSERT_EQ(expected, basicDictionary.getKeys());
}



















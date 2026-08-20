#include "gtest/gtest.h"
#include "rrIniFile.h"
#include <filesystem>
#include <fstream>

using namespace rr;

class IniFileTests : public ::testing::Test {
public:
    std::filesystem::path tempFile;

    IniFileTests() {
        tempFile = std::filesystem::temp_directory_path() / "roadrunner_malformed_ini_test.ini";
    }

    ~IniFileTests() override {
        std::error_code ec;
        std::filesystem::remove(tempFile, ec);
    }

    void writeFile(const std::string &contents) {
        std::ofstream out(tempFile, std::ios::out | std::ios::trunc);
        out << contents;
        out.close();
    }
};

/**
 * Regression tests: Load() and LoadSection() both unconditionally erase()
 * at find_last_of(']') for any line starting with '['. For a section
 * header with no closing bracket, find_last_of returns npos, and
 * erase(npos, 1) throws std::out_of_range -- turning a tolerantly-parsed
 * malformed file into a crash.
 */
TEST_F(IniFileTests, LoadDoesNotThrowOnSectionMissingClosingBracket) {
    writeFile(
        "[missing-close\n"
        "key1=value1\n"
        "[goodsection]\n"
        "key2=value2\n"
    );

    IniFile ini;
    ASSERT_NO_THROW(ini.Load(tempFile.string()));

    // Parsing should have recovered and continued past the malformed line.
    EXPECT_TRUE(ini.SectionExists("goodsection"));
}

TEST_F(IniFileTests, LoadSectionDoesNotThrowOnSectionMissingClosingBracket) {
    writeFile(
        "[missing-close\n"
        "key1=value1\n"
        "[goodsection]\n"
        "key2=value2\n"
    );

    IniFile ini(tempFile.string());
    ASSERT_NO_THROW(ini.LoadSection("goodsection"));
}

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "RoadRunnerTest.h"
#include <filesystem>

using namespace rr;
using namespace std;
using std::filesystem::path;

class CompressedSBML : public RoadRunnerTest {
public:
  path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
  CompressedSBML() = default;

  // Loads the compressed file and checks it matches the uncompressed original.
  void expectSameModel(const string& compressedName) {
    RoadRunner plain((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
    RoadRunner compressed((SBMLFeaturesDir / compressedName).string());
    ASSERT_FALSE(plain.getReactionIds().empty());
    EXPECT_EQ(compressed.getFloatingSpeciesIds(), plain.getFloatingSpeciesIds());
    EXPECT_EQ(compressed.getReactionIds(), plain.getReactionIds());
    EXPECT_EQ(compressed.getStoichiometryIds(), plain.getStoichiometryIds());
  }
};

TEST_F(CompressedSBML, load_bzip2) {
  expectSameModel("named_stoic_in_kinetic_law.xml.bz2");
}

TEST_F(CompressedSBML, load_zip) {
  expectSameModel("named_stoic_in_kinetic_law.xml.zip");
}

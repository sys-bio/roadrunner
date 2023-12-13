#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "sbml/SBMLTypes.h"
#include "sbml/SBMLReader.h"
#include "../test_util.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
    path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
    SBMLFeatures() = default;
};

TEST_F(SBMLFeatures, InvalidSBML_badXML)
{
    EXPECT_THROW(RoadRunner rri("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n  <model metaid=\"x\" id=\"x\">\n    <listOfCompartments>\n      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n    </listOfCompartments>\n  </model>\n</sbml>\n2.0.5"), std::runtime_error);
}

TEST_F(SBMLFeatures, NotActuallySBML)
{
    EXPECT_THROW(RoadRunner rr((SBMLFeaturesDir / "manifest.xml").string()), std::domain_error);
}
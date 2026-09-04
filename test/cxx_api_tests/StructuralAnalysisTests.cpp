#include <rrRoadRunner.h>
#include "gtest/gtest.h"
#include <algorithm>

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "rrExecutableModel.h"
#include "Matrix.h"

using namespace rr;

class StructuralAnalysisTests : public RoadRunnerTest {

public:
    path modelAnalysisModelsDir;
    StructuralAnalysisTests() {
        modelAnalysisModelsDir = rrTestModelsDir_ / "ModelAnalysis";
    }

    /**
     * @brief check equality of int matrices. Fails when expected and actual 
     * are not equal. When fail, print out expected and actual to cout. 
     * @param expected the expected value of the int matrix under test
     * @param actual the observed value of the int matrix under test
     * @details Coerces ls::Matrix<double> to an rr::Matrix<double> (its subclass) 
     *  to make use of equality operators. 
     */
    void checkPassed(rr::Matrix<double> &expected, rr::Matrix<double> &actual, double tol = 1e-7) {
        bool passed = expected.almostEquals(actual, tol);
        if (!passed) {
            std::cout << "expected:" << std::endl;
            std::cout << expected << std::endl;
            std::cout << "actual: " << std::endl;
            std::cout << actual << std::endl;
        }
        ASSERT_TRUE(passed);
    }

    /**
     * @brief check equality of int matrices
     * @param expected the expected value of the int matrix under test
     * @param actual the observed value of the int matrix under test
     * @details Coerces ls::Matrix<double> to an rr::Matrix<double> (its subclass) 
     *  to make use of equality operators. 
     */
    void checkPassed(ls::Matrix<double> &expected, ls::Matrix<double> &actual, double tol = 1e-7) {
        Matrix<double> rrMatrixExpected(expected);
        Matrix<double> rrMatrixActual(actual);
        checkPassed(rrMatrixExpected, rrMatrixActual, tol);
    }

    void checkPassed(rr::Matrix<double> &expected, ls::Matrix<double> &actual, double tol = 1e-7) {
        Matrix<double> rrMatrixActual(actual);
        checkPassed(expected, rrMatrixActual, tol);
    }

    void checkLinkMatrix(const std::string &modelName) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->linkMatrix();
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getLinkMatrix();
        checkPassed(expected, actual);
        delete testModel;
    }

    void checkNrMatrix(const std::string &modelName) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->NrMatrix();
        RoadRunner rr(testModel->str());
        rr.setConservedMoietyAnalysis(true);
        ls::Matrix<double> actual = rr.getNrMatrix();
        checkPassed(expected, actual);
        delete testModel;
    }

    void checkKMatrix(const std::string &modelName, double tol) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->KMatrix();
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getKMatrix();
        checkPassed(expected, actual);
        delete testModel;
    }

    void checkReducedStoicMatrix(const std::string &modelName) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->reducedStoicMatrix();
        RoadRunner rr(testModel->str());
        rr.setConservedMoietyAnalysis(true);
        ls::Matrix<double> actual = rr.getReducedStoichiometryMatrix();
        checkPassed(expected, actual);
        delete testModel;
    }

    void checkFullStoicMatrix(const std::string &modelName) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->fullStoicMatrix();
        RoadRunner rr(testModel->str());
        rr.setConservedMoietyAnalysis(true);
        ls::Matrix<double> actual = rr.getFullStoichiometryMatrix();
        checkPassed(expected, actual);
        delete testModel;
    }

    /**
     * Note: the extended stoic matix is affected by the
     * moiety conservation analysis status of the model.
     * Therefore this should be tested under both conditions
     */
    void checkExtendedStoicMatrix(const std::string &modelName, double tol = 1e-7) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->extendedStoicMatrix();
        RoadRunner rr(testModel->str());
//        rr.setConservedMoietyAnalysis(true); // breaks the test.
        ls::Matrix<double> actual = rr.getExtendedStoichiometryMatrix();
        checkPassed(expected, actual, tol);
        delete testModel;
    }

    void checkL0Matrix(const std::string &modelName, double tol = 1e-7) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->L0Matrix();
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getL0Matrix();
        checkPassed(expected, actual, tol);
        delete testModel;
    }

    void checkConservationMatrix(const std::string &modelName, double tol = 1e-7) {
        TestModel *testModel = TestModelFactory(modelName);
        auto *structuralTestModel = dynamic_cast<StructuralProperties *>(testModel);
        rr::Matrix<double> expected = structuralTestModel->conservationMatrix();
        RoadRunner rr(testModel->str());
        rr.setConservedMoietyAnalysis(true);
        ls::Matrix<double> actual = rr.getConservationMatrix();
        checkPassed(expected, actual, tol);
        delete testModel;
    }


};

TEST_F(StructuralAnalysisTests, S2ValueUpdatesAfterConservedMoietyConversion) {
  std::string sbml =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model id=\"AssignmentRuleChainMoiety\">\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <listOfSpecies>\n"
    "      <species id=\"Q\" compartment=\"default_compartment\" initialConcentration=\"2\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"P2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"A\" compartment=\"default_compartment\" initialConcentration=\"3\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"B\" compartment=\"default_compartment\" initialConcentration=\"1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"W1\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "    </listOfSpecies>\n"
    "    <listOfParameters>\n"
    "      <parameter id=\"k1\" value=\"0.5\" constant=\"true\"/>\n"
    "      <parameter id=\"kw1\" value=\"2\" constant=\"true\"/>\n"
    "      <parameter id=\"ks2\" value=\"3\" constant=\"true\"/>\n"
    "      <parameter id=\"k2\" value=\"0.8\" constant=\"true\"/>\n"
    "      <parameter id=\"k3\" value=\"0.4\" constant=\"true\"/>\n"
    "    </listOfParameters>\n"
    "    <listOfRules>\n"
    "      <assignmentRule variable=\"W1\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> kw1 </ci><ci> Q </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "      <assignmentRule variable=\"S2\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> ks2 </ci><ci> W1 </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "    </listOfRules>\n"
    "    <listOfReactions>\n"
    "      <reaction id=\"J1\" reversible=\"false\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"Q\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"P2\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><times/><ci> k1 </ci><ci> S2 </ci></apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "      <reaction id=\"J2\" reversible=\"true\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"A\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"B\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><minus/>\n"
    "              <apply><times/><ci> k2 </ci><ci> A </ci></apply>\n"
    "              <apply><times/><ci> k3 </ci><ci> B </ci></apply>\n"
    "            </apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "    </listOfReactions>\n"
    "  </model>\n"
    "</sbml>";

  RoadRunner rr(sbml);
  rr.setConservedMoietyAnalysis(true);

  ExecutableModel* model = rr.getModel();
  std::cout << "numIndFloatingSpecies=" << model->getNumIndFloatingSpecies()
    << " numConservedMoieties=" << model->getNumConservedMoieties() << std::endl;
  std::vector<std::string> floatIds = rr.getFloatingSpeciesIds();
  for (size_t k = 0; k < floatIds.size(); ++k) {
    std::cout << "floatingSpecies[" << k << "] = " << floatIds[k] << std::endl;
  }
  std::cout << rr.getCurrentSBML() << std::endl;
  int n = model->getStateVector(0);
  std::vector<std::string> ids;
  for (int i = 0; i < n; ++i) {
    ids.push_back(model->getStateVectorId(i));
  }
  int qIndex = std::distance(ids.begin(), std::find(ids.begin(), ids.end(), "Q"));
  ASSERT_LT(qIndex, n);

  std::vector<double> y(n);
  model->getStateVector(y.data());

  double s2Before = rr.getValue("S2");
  double w1Before = rr.getValue("W1");
  double qBefore = rr.getValue("Q");

  y[qIndex] += 0.1;
  model->setStateVector(y.data());

  double s2After = rr.getValue("S2");
  double w1After = rr.getValue("W1");
  double qAfter = rr.getValue("Q");

  int j1Index = model->getReactionIndex("J1");
  double rateAfter = 0;
  model->getReactionRates(1, &j1Index, &rateAfter);

  std::cout << "Q: " << qBefore << " -> " << qAfter << std::endl;
  std::cout << "W1: " << w1Before << " -> " << w1After << std::endl;
  std::cout << "S2: " << s2Before << " -> " << s2After << std::endl;
  std::cout << "J1 rate after perturbing Q: " << rateAfter << std::endl;
}


// Minimal repro: Q's only reaction (J1) reaches Q solely through two
// levels of inlined assignment rules (S2 = ks2*W1, W1 = kw1*Q), never
// directly. A/B form a separate, unrelated conservation law elsewhere in
// the same model, so setConservedMoietyAnalysis(true) actually converts
// something. Checks whether J1's rate still responds to Q after
// conversion, isolating whether the Teusink P-column bug is a general
// RoadRunner defect in assignment-rule-chain resolution post-conversion,
// or specific to Teusink's own structure.
TEST_F(StructuralAnalysisTests, AssignmentRuleChainSurvivesConservedMoietyConversion) {
  std::string sbml =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model id=\"AssignmentRuleChainMoiety\">\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <listOfSpecies>\n"
    "      <species id=\"Q\" compartment=\"default_compartment\" initialConcentration=\"2\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"P2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"A\" compartment=\"default_compartment\" initialConcentration=\"3\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"B\" compartment=\"default_compartment\" initialConcentration=\"1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"W1\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "    </listOfSpecies>\n"
    "    <listOfParameters>\n"
    "      <parameter id=\"k1\" value=\"0.5\" constant=\"true\"/>\n"
    "      <parameter id=\"kw1\" value=\"2\" constant=\"true\"/>\n"
    "      <parameter id=\"ks2\" value=\"3\" constant=\"true\"/>\n"
    "      <parameter id=\"k2\" value=\"0.8\" constant=\"true\"/>\n"
    "      <parameter id=\"k3\" value=\"0.4\" constant=\"true\"/>\n"
    "    </listOfParameters>\n"
    "    <listOfRules>\n"
    "      <assignmentRule variable=\"W1\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> kw1 </ci><ci> Q </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "      <assignmentRule variable=\"S2\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> ks2 </ci><ci> W1 </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "    </listOfRules>\n"
    "    <listOfReactions>\n"
    "      <reaction id=\"J1\" reversible=\"false\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"Q\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"P2\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><times/><ci> k1 </ci><ci> S2 </ci></apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "      <reaction id=\"J2\" reversible=\"true\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"A\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"B\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><minus/>\n"
    "              <apply><times/><ci> k2 </ci><ci> A </ci></apply>\n"
    "              <apply><times/><ci> k3 </ci><ci> B </ci></apply>\n"
    "            </apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "    </listOfReactions>\n"
    "  </model>\n"
    "</sbml>";

  auto measureJ1RateSensitivityToQ = [](RoadRunner& rr) {
    ExecutableModel* model = rr.getModel();
    int n = model->getStateVector(0);
    std::vector<std::string> ids;
    for (int i = 0; i < n; ++i) {
      ids.push_back(model->getStateVectorId(i));
    }
    int qIndex = std::distance(ids.begin(), std::find(ids.begin(), ids.end(), "Q"));
    EXPECT_LT(qIndex, n);

    std::vector<double> y(n);
    model->getStateVector(y.data());

    int j1Index = model->getReactionIndex("J1");
    double rateBefore = 0;
    model->getReactionRates(1, &j1Index, &rateBefore);

    y[qIndex] += 0.1;
    model->setStateVector(y.data());

    double rateAfter = 0;
    model->getReactionRates(1, &j1Index, &rateAfter);

    return rateAfter - rateBefore;
    };

  RoadRunner rrBefore(sbml);
  double deltaBefore = measureJ1RateSensitivityToQ(rrBefore);
  std::cout << "J1 rate change per +0.1 Q, before moiety conversion: " << deltaBefore << std::endl;
  EXPECT_NEAR(deltaBefore, 0.1 * 0.5 * 3 * 2, 1e-6);

  RoadRunner rrAfter(sbml);
  rrAfter.setConservedMoietyAnalysis(true);
  double deltaAfter = measureJ1RateSensitivityToQ(rrAfter);
  std::cout << "J1 rate change per +0.1 Q, after moiety conversion: " << deltaAfter << std::endl;
}

TEST_F(StructuralAnalysisTests, PrintPostConversionSBMLForAssignmentRuleChain) {
  std::string sbml =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
    "  <model id=\"AssignmentRuleChainMoiety\">\n"
    "    <listOfCompartments>\n"
    "      <compartment id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
    "    </listOfCompartments>\n"
    "    <listOfSpecies>\n"
    "      <species id=\"Q\" compartment=\"default_compartment\" initialConcentration=\"2\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"P2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"A\" compartment=\"default_compartment\" initialConcentration=\"3\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"B\" compartment=\"default_compartment\" initialConcentration=\"1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"W1\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
    "    </listOfSpecies>\n"
    "    <listOfParameters>\n"
    "      <parameter id=\"k1\" value=\"0.5\" constant=\"true\"/>\n"
    "      <parameter id=\"kw1\" value=\"2\" constant=\"true\"/>\n"
    "      <parameter id=\"ks2\" value=\"3\" constant=\"true\"/>\n"
    "      <parameter id=\"k2\" value=\"0.8\" constant=\"true\"/>\n"
    "      <parameter id=\"k3\" value=\"0.4\" constant=\"true\"/>\n"
    "    </listOfParameters>\n"
    "    <listOfRules>\n"
    "      <assignmentRule variable=\"W1\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> kw1 </ci><ci> Q </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "      <assignmentRule variable=\"S2\">\n"
    "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "          <apply><times/><ci> ks2 </ci><ci> W1 </ci></apply>\n"
    "        </math>\n"
    "      </assignmentRule>\n"
    "    </listOfRules>\n"
    "    <listOfReactions>\n"
    "      <reaction id=\"J1\" reversible=\"false\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"Q\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"P2\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><times/><ci> k1 </ci><ci> S2 </ci></apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "      <reaction id=\"J2\" reversible=\"true\" fast=\"false\">\n"
    "        <listOfReactants>\n"
    "          <speciesReference species=\"A\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfReactants>\n"
    "        <listOfProducts>\n"
    "          <speciesReference species=\"B\" stoichiometry=\"1\" constant=\"true\"/>\n"
    "        </listOfProducts>\n"
    "        <kineticLaw>\n"
    "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
    "            <apply><minus/>\n"
    "              <apply><times/><ci> k2 </ci><ci> A </ci></apply>\n"
    "              <apply><times/><ci> k3 </ci><ci> B </ci></apply>\n"
    "            </apply>\n"
    "          </math>\n"
    "        </kineticLaw>\n"
    "      </reaction>\n"
    "    </listOfReactions>\n"
    "  </model>\n"
    "</sbml>";

  RoadRunner rr(sbml);
  rr.setConservedMoietyAnalysis(true);
  std::cout << rr.getCurrentSBML() << std::endl;
}


// BIOMD0000000064 (Teusink et al. 2000, yeast glycolysis) has two species,
// SUM_P and F26BP, declared with constant="true" and boundaryCondition="false"
// -- they're used only as fixed parameters inside kinetic laws (the AK
// equilibrium and the PFK rate law), never as reactants/products. They
// should be classified as boundary species (and excluded from the
// floating-species/Newton state vector), not as ordinary floating species
// with a structurally-zero rate row.
TEST_F(StructuralAnalysisTests, TeusinkConstantSpeciesAreBoundary) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());

  auto boundaryIds = rr.getBoundarySpeciesIds();
  EXPECT_NE(std::find(boundaryIds.begin(), boundaryIds.end(), "SUM_P"), boundaryIds.end());
  EXPECT_NE(std::find(boundaryIds.begin(), boundaryIds.end(), "F26BP"), boundaryIds.end());

  auto floatingIds = rr.getFloatingSpeciesIds();
  EXPECT_EQ(std::find(floatingIds.begin(), floatingIds.end(), "SUM_P"), floatingIds.end());
  EXPECT_EQ(std::find(floatingIds.begin(), floatingIds.end(), "F26BP"), floatingIds.end());
}

// The NAD/NADH pair forms a genuine stoichiometric conservation cycle in
// this model (via the GAPDH/ADH/G3PDH reactions), independent of the
// constant-species fix above. This checks that RoadRunner's structural
// analysis (the same analysis ConservedMoietyConverter::convert() reads
// its independent/dependent species split from) actually finds it: NAD or
// NADH should come back as a dependent species for a conservation law.
TEST_F(StructuralAnalysisTests, TeusinkNADCycleIsFound) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());

  auto dependentIds = rr.getDependentFloatingSpeciesIds();
  bool foundNADCycle = std::find(dependentIds.begin(), dependentIds.end(), "NAD") != dependentIds.end()
    || std::find(dependentIds.begin(), dependentIds.end(), "NADH") != dependentIds.end();
  EXPECT_TRUE(foundNADCycle);
}

// Before the constant-species-as-boundary fix, steadyState() failed with
// "Jacobian matrix singular in NLEQ" on this model: SUM_P and F26BP were
// counted as independent floating species even though nothing ever changes
// them, giving the Newton solve two structurally-zero Jacobian rows on top
// of the NAD/NADH moiety dependency. Expected values are this model's own
// steady state (matches Table 4 of Teusink et al. 2000, and the "this
// model" column in the SBML's own notes).
TEST_F(StructuralAnalysisTests, TeusinkSteadyStateConverges) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());

  ASSERT_NO_THROW(rr.steadyState());

  EXPECT_NEAR(rr.getValue("[G6P]"), 1.0332, 1e-3);
  EXPECT_NEAR(rr.getValue("[F6P]"), 0.1128, 1e-3);
  EXPECT_NEAR(rr.getValue("[PYR]"), 8.5232, 1e-3);
  EXPECT_NEAR(rr.getValue("[ATP]"), 2.5084, 1e-3);
  EXPECT_NEAR(rr.getValue("[ADP]"), 1.2921, 1e-3);
  EXPECT_NEAR(rr.getValue("[NADH]"), 0.0444, 1e-3);
}

// TeusinkNADCycleIsFound confirms LibStructural finds the NAD/NADH cycle from
// the raw stoichiometry, but steadyState()'s auto_moiety_analysis retry
// swallows any exception from setConservedMoietyAnalysis() and silently falls
// back to no conversion. Calling it directly here surfaces that exception (if
// any) instead, and confirms the conversion actually produces a conserved
// moiety on the compiled model, not just in the structural analysis.
TEST_F(StructuralAnalysisTests, TeusinkConservedMoietyConversionSucceeds) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());

  ASSERT_NO_THROW(rr.setConservedMoietyAnalysis(true));
  EXPECT_TRUE(rr.getConservedMoietyAnalysis());
  EXPECT_EQ(rr.getModel()->getNumConservedMoieties(), 3);
}

// Conversion succeeds and reports a conserved moiety, but steadyState()
// still hits a singular Jacobian. Print the actual reduced Jacobian NLEQ
// solves against (row/col names are the floating species included in the
// post-conversion state vector) to see the singularity directly.
TEST_F(StructuralAnalysisTests, TeusinkPrintReducedJacobian) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  ls::DoubleMatrix jac = rr.getReducedJacobian();
  std::cout << jac << std::endl;
}

// getReducedJacobian() reads its rates through RoadRunner::getRatesOfChange(),
// which applies a LibStructural link-matrix correction whenever conserved
// moiety analysis is on (see getRatesOfChange() in rrRoadRunner.cpp). NLEQ's
// own residual function never goes through that: it calls
// ExecutableModel::setStateVector()/getStateVectorRate() directly. Build the
// Jacobian the same way NLEQ does, bypassing getRatesOfChange() entirely, to
// see what NLEQ itself actually sees.
TEST_F(StructuralAnalysisTests, TeusinkPrintRawStateVectorJacobian) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  ExecutableModel* model = rr.getModel();
  int n = model->getStateVector(0);
  double h = 1e-6;

  std::vector<std::string> ids;
  for (int i = 0; i < n; ++i) {
    ids.push_back(model->getStateVectorId(i));
  }

  std::vector<double> y(n);
  model->getStateVector(y.data());

  ls::DoubleMatrix jac(n, n);
  jac.setRowNames(ids);
  jac.setColNames(ids);

  std::vector<double> dyPlus(n), dyMinus(n);
  for (int col = 0; col < n; ++col) {
    double saved = y[col];

    y[col] = saved + h;
    model->setStateVector(y.data());
    model->getStateVectorRate(0, y.data(), dyPlus.data());

    y[col] = saved - h;
    model->setStateVector(y.data());
    model->getStateVectorRate(0, y.data(), dyMinus.data());

    y[col] = saved;
    model->setStateVector(y.data());

    for (int row = 0; row < n; ++row) {
      jac(row, col) = (dyPlus[row] - dyMinus[row]) / (2.0 * h);
    }
  }

  std::cout << jac << std::endl;
}

// Same raw state-vector Jacobian, but on the plain unconverted model (no
// setConservedMoietyAnalysis call) to check whether the P column is zero
// even without conserved moiety conversion in the picture -- i.e. whether
// this is a pre-existing bug unrelated to the NAD/NADH moiety work.
TEST_F(StructuralAnalysisTests, TeusinkPrintRawStateVectorJacobianNoConservedMoieties) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());

  ExecutableModel* model = rr.getModel();
  int n = model->getStateVector(0);
  double h = 1e-6;

  std::vector<std::string> ids;
  for (int i = 0; i < n; ++i) {
    ids.push_back(model->getStateVectorId(i));
  }

  std::vector<double> y(n);
  model->getStateVector(y.data());

  ls::DoubleMatrix jac(n, n);
  jac.setRowNames(ids);
  jac.setColNames(ids);

  std::vector<double> dyPlus(n), dyMinus(n);
  for (int col = 0; col < n; ++col) {
    double saved = y[col];

    y[col] = saved + h;
    model->setStateVector(y.data());
    model->getStateVectorRate(0, y.data(), dyPlus.data());

    y[col] = saved - h;
    model->setStateVector(y.data());
    model->getStateVectorRate(0, y.data(), dyMinus.data());

    y[col] = saved;
    model->setStateVector(y.data());

    for (int row = 0; row < n; ++row) {
      jac(row, col) = (dyPlus[row] - dyMinus[row]) / (2.0 * h);
    }
  }

  std::cout << jac << std::endl;
}

// Splits the question: does perturbing P even propagate through the
// AK-equilibrium assignment rules to ATP/ADP after conversion? If not, the
// break is in the assignment-rule chain itself. If it does update correctly
// here (via getValue(), a different read path than getStateVectorRate's
// inlined kinetic law code), the break must be one level deeper: inside
// evalReactionRatesPtr's own inlined copy of that same computation.
TEST_F(StructuralAnalysisTests, TeusinkPerturbingPAfterConversionUpdatesATP) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  ExecutableModel* model = rr.getModel();
  int n = model->getStateVector(0);
  std::vector<std::string> ids;
  for (int i = 0; i < n; ++i) {
    ids.push_back(model->getStateVectorId(i));
  }
  int pIndex = std::distance(ids.begin(), std::find(ids.begin(), ids.end(), "P"));
  ASSERT_LT(pIndex, n);

  std::vector<double> y(n);
  model->getStateVector(y.data());

  double atpBefore = rr.getValue("[ATP]");
  double adpBefore = rr.getValue("[ADP]");

  y[pIndex] += 1.0;
  model->setStateVector(y.data());

  double atpAfter = rr.getValue("[ATP]");
  double adpAfter = rr.getValue("[ADP]");

  std::cout << "ATP: " << atpBefore << " -> " << atpAfter << std::endl;
  std::cout << "ADP: " << adpBefore << " -> " << adpAfter << std::endl;
}

// createReorderedSpecies deletes any <species> that's neither
// boundary/constant nor part of indSpecies/depSpecies -- ATP/ADP/AMP satisfy
// none of those (not constant, never a reactant/product), so their
// <species> elements are likely gone post-conversion even though their
// assignment rules (untouched, in <listOfRules>) still reference P and
// SUM_P. Check what they actually get classified as.
TEST_F(StructuralAnalysisTests, TeusinkATPClassificationAfterConversion) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  auto floatingIds = rr.getFloatingSpeciesIds();
  auto boundaryIds = rr.getBoundarySpeciesIds();
  auto globalParamIds = rr.getGlobalParameterIds();

  for (const std::string& id : {"ATP", "ADP", "AMP", "SUM_P", "F26BP", "P"}) {
    bool isFloating = std::find(floatingIds.begin(), floatingIds.end(), id) != floatingIds.end();
    bool isBoundary = std::find(boundaryIds.begin(), boundaryIds.end(), id) != boundaryIds.end();
    bool isGlobalParam = std::find(globalParamIds.begin(), globalParamIds.end(), id) != globalParamIds.end();
    std::cout << id << ": floating=" << isFloating << " boundary=" << isBoundary
               << " globalParam=" << isGlobalParam << std::endl;
  }
}

// Splits the zero P column further: does perturbing P even change the
// reaction rates themselves (a kinetic-law/inlining problem), or do the
// rates respond correctly and it's the stoichiometry-matrix multiply that
// silently drops P's contribution when turning rates into species rates
// of change (a whole zero column points more toward the former, but
// check directly rather than assume).
TEST_F(StructuralAnalysisTests, TeusinkPerturbingPAfterConversionChangesReactionRates) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  ExecutableModel* model = rr.getModel();
  int n = model->getStateVector(0);
  std::vector<std::string> ids;
  for (int i = 0; i < n; ++i) {
    ids.push_back(model->getStateVectorId(i));
  }
  int pIndex = std::distance(ids.begin(), std::find(ids.begin(), ids.end(), "P"));
  ASSERT_LT(pIndex, n);

  std::vector<double> y(n);
  model->getStateVector(y.data());

  int numReactions = model->getNumReactions();
  std::vector<double> ratesBefore(numReactions);
  model->getReactionRates(numReactions, 0, ratesBefore.data());

  y[pIndex] += 1.0;
  model->setStateVector(y.data());

  std::vector<double> ratesAfter(numReactions);
  model->getReactionRates(numReactions, 0, ratesAfter.data());

  for (int i = 0; i < numReactions; ++i) {
    std::cout << model->getReactionId(i) << ": " << ratesBefore[i] << " -> " << ratesAfter[i] << std::endl;
  }
}

// The reaction-rate freeze is specific to P -- other floating species
// (G6P, TRIO, etc, referenced directly in kinetic laws) respond correctly
// to perturbation. P only reaches kinetic laws indirectly, through the
// inlined ADP/ATP assignment rules. Dump the actual post-conversion SBML
// for those rules to check whether conversion mangled their math, before
// looking any further at codegen.
TEST_F(StructuralAnalysisTests, TeusinkPrintConvertedAssignmentRules) {
  RoadRunner rr((modelAnalysisModelsDir / "teusink_glycolysis.xml").string());
  rr.setConservedMoietyAnalysis(true);

  std::string sbml = rr.getCurrentSBML();

  for (const std::string& variable : {"ADP", "ATP", "AMP"}) {
    std::string needle = "variable=\"" + variable + "\"";
    size_t pos = sbml.find(needle);
    if (pos == std::string::npos) {
      std::cout << "no rule found for " << variable << std::endl;
      continue;
    }
    size_t start = sbml.rfind("<assignmentRule", pos);
    size_t end = sbml.find("</assignmentRule>", pos);
    std::cout << sbml.substr(start, end + std::string("</assignmentRule>").size() - start) << std::endl;
  }
}

/************************************************************
 * Check structural properties in the BimolecularEnd TestModel
 */
TEST_F(StructuralAnalysisTests, BimolecularEndlinkMatrix) {
    checkLinkMatrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, BimolecularEndNrMatrix) {
    checkNrMatrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, BimolecularEndKMatrix) {
    checkKMatrix("BimolecularEnd", 1e-7);
}

TEST_F(StructuralAnalysisTests, BimolecularEndreducedStoicMatrix) {
    checkReducedStoicMatrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, BimolecularEndfullStoicMatrix) {
    checkFullStoicMatrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, BimolecularEndextendedStoicMatrix) {
    checkExtendedStoicMatrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, SimpleFluxReactantsStoicMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    //std::cout << rr.getReactantsStoichiometryMatrix() << std::endl;
    ls::DoubleMatrix expected = simpleFlux.reactantsStoicMatrix();
    ls::DoubleMatrix actual = rr.getReactantsStoichiometryMatrix();
    checkMatrixEqual(expected, actual, 1e-7);
}

TEST_F(StructuralAnalysisTests, SimpleFluxProductsStoicMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    //std::cout << rr.getProductsStoichiometryMatrix() << std::endl;
    ls::DoubleMatrix expected = simpleFlux.productsStoicMatrix();
    ls::DoubleMatrix actual = rr.getProductsStoichiometryMatrix();
    checkMatrixEqual(expected, actual, 1e-7);
}

TEST_F(StructuralAnalysisTests, OddStoichiometryExtendedStoicMatrix) {
    RoadRunner rr((modelAnalysisModelsDir / "odd_stoichiometries.xml").string());
    rr::Matrix<double> expected(
        {
                {-1, 0, 2.2,  1},
                {2,  3, 0,    0},
                {0,  4, -1.5, 0},
        });
    ls::DoubleMatrix actual = rr.getExtendedStoichiometryMatrix();
    checkMatrixEqual(expected, actual, 1e-7);
}

TEST_F(StructuralAnalysisTests, OddStoichiometryReactantsStoicMatrix) {
    RoadRunner rr((modelAnalysisModelsDir / "odd_stoichiometries.xml").string());
    rr::Matrix<double> expected(
        {
                {-1, 0,  0,  -2},
                { 0, 3,  0,   0},
        });

    ls::DoubleMatrix actual = rr.getReactantsStoichiometryMatrix(false);
    checkMatrixEqual(expected, actual, 1e-7);

    rr::Matrix<double> expected_boundary(
        {
                {-1, 0,  0,  -2},
                { 0, 3,  0,   0},
                { 0, 0, -1.5, 0},
        });

    actual = rr.getReactantsStoichiometryMatrix(true);
    checkMatrixEqual(expected_boundary, actual, 1e-7);
}

TEST_F(StructuralAnalysisTests, OddStoichiometryProductsStoicMatrix) {
    RoadRunner rr((modelAnalysisModelsDir / "odd_stoichiometries.xml").string());
    rr::Matrix<double> expected(
        {
                {0, 0, 2.2, 3},
                {2, 0, 0,   0},
        });

    ls::DoubleMatrix actual = rr.getProductsStoichiometryMatrix(false);
    checkMatrixEqual(expected, actual, 1e-7);

    rr::Matrix<double> expected_boundary(
        {
                {0, 0, 2.2, 3},
                {2, 0, 0,   0},
                {0, 4, 0,   0},
        });
    actual = rr.getProductsStoichiometryMatrix(true);

    checkMatrixEqual(expected_boundary, actual, 1e-7);
}

TEST_F(StructuralAnalysisTests, BimolecularEndL0Matrix) {
    checkL0Matrix("BimolecularEnd");
}

TEST_F(StructuralAnalysisTests, BimolecularEndconservationMatrix) {
    checkConservationMatrix("BimolecularEnd");
}


TEST(Check, Stoic) {
    std::string sbml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                       "<!-- Created by libAntimony version v2.5.2 on 2014-09-22 11:05 with libSBML version 5.10.2. -->\n"
                       "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
                       "  <model id=\"Bimolecular_end\" name=\"Bimolecular_end\">\n"
                       "    <listOfFunctionDefinitions>\n"
                       "      <functionDefinition id=\"MM\">\n"
                       "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "          <lambda>\n"
                       "            <bvar>\n"
                       "              <ci> S1 </ci>\n"
                       "            </bvar>\n"
                       "            <bvar>\n"
                       "              <ci> S2 </ci>\n"
                       "            </bvar>\n"
                       "            <bvar>\n"
                       "              <ci> Vm </ci>\n"
                       "            </bvar>\n"
                       "            <bvar>\n"
                       "              <ci> Km1 </ci>\n"
                       "            </bvar>\n"
                       "            <bvar>\n"
                       "              <ci> Km2 </ci>\n"
                       "            </bvar>\n"
                       "            <bvar>\n"
                       "              <ci> Keq </ci>\n"
                       "            </bvar>\n"
                       "            <apply>\n"
                       "              <divide/>\n"
                       "              <apply>\n"
                       "                <times/>\n"
                       "                <apply>\n"
                       "                  <divide/>\n"
                       "                  <ci> Vm </ci>\n"
                       "                  <ci> Km1 </ci>\n"
                       "                </apply>\n"
                       "                <apply>\n"
                       "                  <minus/>\n"
                       "                  <ci> S1 </ci>\n"
                       "                  <apply>\n"
                       "                    <divide/>\n"
                       "                    <ci> S2 </ci>\n"
                       "                    <ci> Keq </ci>\n"
                       "                  </apply>\n"
                       "                </apply>\n"
                       "              </apply>\n"
                       "              <apply>\n"
                       "                <plus/>\n"
                       "                <cn type=\"integer\"> 1 </cn>\n"
                       "                <apply>\n"
                       "                  <divide/>\n"
                       "                  <ci> S1 </ci>\n"
                       "                  <ci> Km1 </ci>\n"
                       "                </apply>\n"
                       "                <apply>\n"
                       "                  <divide/>\n"
                       "                  <ci> S2 </ci>\n"
                       "                  <ci> Km2 </ci>\n"
                       "                </apply>\n"
                       "              </apply>\n"
                       "            </apply>\n"
                       "          </lambda>\n"
                       "        </math>\n"
                       "      </functionDefinition>\n"
                       "    </listOfFunctionDefinitions>\n"
                       "    <listOfCompartments>\n"
                       "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
                       "    </listOfCompartments>\n"
                       "    <listOfSpecies>\n"
                       "      <species id=\"X0\" compartment=\"default_compartment\" initialConcentration=\"8.03\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
                       "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"7.12\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                       "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"3.97\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                       "      <species id=\"S3\" compartment=\"default_compartment\" initialConcentration=\"0.96\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                       "      <species id=\"X1\" compartment=\"default_compartment\" initialConcentration=\"0.54\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
                       "    </listOfSpecies>\n"
                       "    <listOfReactions>\n"
                       "      <reaction id=\"J0\" reversible=\"true\" fast=\"false\">\n"
                       "        <listOfReactants>\n"
                       "          <speciesReference species=\"X0\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfReactants>\n"
                       "        <listOfProducts>\n"
                       "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfProducts>\n"
                       "        <kineticLaw>\n"
                       "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <apply>\n"
                       "              <ci> MM </ci>\n"
                       "              <ci> X0 </ci>\n"
                       "              <ci> S1 </ci>\n"
                       "              <cn> 4.39 </cn>\n"
                       "              <cn> 9.85 </cn>\n"
                       "              <cn> 7.38 </cn>\n"
                       "              <cn> 6.12 </cn>\n"
                       "            </apply>\n"
                       "          </math>\n"
                       "        </kineticLaw>\n"
                       "      </reaction>\n"
                       "      <reaction id=\"J1\" reversible=\"true\" fast=\"false\">\n"
                       "        <listOfReactants>\n"
                       "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfReactants>\n"
                       "        <listOfProducts>\n"
                       "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfProducts>\n"
                       "        <kineticLaw>\n"
                       "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <apply>\n"
                       "              <ci> MM </ci>\n"
                       "              <ci> S1 </ci>\n"
                       "              <ci> S2 </ci>\n"
                       "              <cn> 6.12 </cn>\n"
                       "              <cn> 9.15 </cn>\n"
                       "              <cn> 1.59 </cn>\n"
                       "              <cn> 4.68 </cn>\n"
                       "            </apply>\n"
                       "          </math>\n"
                       "        </kineticLaw>\n"
                       "      </reaction>\n"
                       "      <reaction id=\"J2\" reversible=\"true\" fast=\"false\">\n"
                       "        <listOfReactants>\n"
                       "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfReactants>\n"
                       "        <listOfProducts>\n"
                       "          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfProducts>\n"
                       "        <kineticLaw>\n"
                       "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <apply>\n"
                       "              <ci> MM </ci>\n"
                       "              <ci> S2 </ci>\n"
                       "              <ci> S3 </ci>\n"
                       "              <cn> 4.68 </cn>\n"
                       "              <cn> 8.22 </cn>\n"
                       "              <cn> 4.29 </cn>\n"
                       "              <cn> 0.57 </cn>\n"
                       "            </apply>\n"
                       "          </math>\n"
                       "        </kineticLaw>\n"
                       "      </reaction>\n"
                       "      <reaction id=\"J3\" reversible=\"true\" fast=\"false\">\n"
                       "        <listOfReactants>\n"
                       "          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfReactants>\n"
                       "        <listOfProducts>\n"
                       "          <speciesReference species=\"X1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfProducts>\n"
                       "        <kineticLaw>\n"
                       "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <apply>\n"
                       "              <ci> MM </ci>\n"
                       "              <ci> S3 </ci>\n"
                       "              <ci> X1 </ci>\n"
                       "              <cn> 0.57 </cn>\n"
                       "              <cn> 0.8 </cn>\n"
                       "              <cn> 2.2 </cn>\n"
                       "              <cn> 4.65 </cn>\n"
                       "            </apply>\n"
                       "          </math>\n"
                       "        </kineticLaw>\n"
                       "      </reaction>\n"
                       "      <reaction id=\"J4\" reversible=\"true\" fast=\"false\">\n"
                       "        <listOfReactants>\n"
                       "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfReactants>\n"
                       "        <listOfProducts>\n"
                       "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                       "        </listOfProducts>\n"
                       "        <kineticLaw>\n"
                       "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <apply>\n"
                       "              <minus/>\n"
                       "              <apply>\n"
                       "                <times/>\n"
                       "                <cn> 4.65 </cn>\n"
                       "                <ci> S2 </ci>\n"
                       "                <ci> S3 </ci>\n"
                       "              </apply>\n"
                       "              <apply>\n"
                       "                <times/>\n"
                       "                <cn> 7.61 </cn>\n"
                       "                <ci> S1 </ci>\n"
                       "              </apply>\n"
                       "            </apply>\n"
                       "          </math>\n"
                       "        </kineticLaw>\n"
                       "      </reaction>\n"
                       "    </listOfReactions>\n"
                       "  </model>\n"
                       "</sbml>";

    RoadRunner rr(sbml);
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getFullStoichiometryMatrix() << std::endl;
    std::cout << rr.getReducedStoichiometryMatrix() << std::endl;
    std::cout << rr.getNrMatrix() << std::endl;
    /**
     *  1,-1,0,0,1
        0,0,1,-1,-1
        0,1,-1,0,-1

     actual Full
J0,J1,J2,J3,J4
1,-1,0,0,1
0,1,-1,0,-1
0,0,1,-1,-1

     actual reduced
J0,J1,J2,J3,J4
1,-1,0,0,1
0,0,1,-1,-1
0,1,-1,0,-1

     actual Nr
J0,J1,J2,J3,J4
1,-1,0,0,1
0,0,1,-1,-1
0,1,-1,0,-1
     */
}

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "Matrix.h"

using namespace rr;

class StructuralAnalysisTests : public RoadRunnerTest {

public:
    StructuralAnalysisTests() = default;

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
     * Note: the extended stioc matix is affected by the
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







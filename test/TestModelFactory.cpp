#include "gtest/gtest.h"
#include <fstream>
#include "TestModelFactory.h"

std::vector<int> getVofI(){
    return std::vector<int>({1, 2, 3});
}

#include "SteadyStateSolver.h"
#include "Matrix.h"

void TestModel::toFile(const std::string &fname) {
    std::ofstream of(fname);
    of << str();
    of.close();
}

std::unordered_map<std::string, rr::Setting> SteadyStateResult::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>();
}

void SteadyStateResult::applySteadyStateSettings(rr::RoadRunner *rr) {
    for (auto &settingsIterator : steadyStateSettings()) {
        if (settingsIterator.first == "moiety_conservation") {
            rr->setConservedMoietyAnalysis(settingsIterator.second);
        } else {
            try {
                rr->getSteadyStateSolver()->setValue(settingsIterator.first, settingsIterator.second);
            } catch (std::exception &err) {
                // if solver does not have this option, that's okay (since we deal with nleq and sundials together
                continue;
            }
        }
    }
}

void SteadyStateResult::checkSteadyState(rr::RoadRunner *rr, double tol) {
    applySteadyStateSettings(rr);
    rr->steadyState();
    // collect actual results from model
    auto result = rr->getFloatingSpeciesConcentrationsNamedArray();
    std::vector<std::string> names = result.getColNames();

    auto expectedResult = steadyState();

    // check to see if actual result are near expected.
    for (int i = 0; i < names.size(); i++) {
        std::string speciesID = names[i];
        double actualResult = result[0][i]; // 0th row, ith col of a DoubleMatrix
        double expected = expectedResult[speciesID]; // first is start val, second is speciesID at steady state

        std::cout << "Comparing \"" << speciesID << "\" expected result: " << expected
                  << " with actual result " << actualResult << std::endl;
        EXPECT_NEAR(expected, actualResult, tol);
    }
}

std::unordered_map<std::string, rr::Setting> TimeSeriesResult::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>();
}

std::string SimpleFlux::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"x\" id=\"x\" name=\"SimpleFlux\" >\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"10\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"kf\" value=\"0.1\" constant=\"true\"/>\n"
           "      <parameter id=\"kb\" value=\"0.01\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"_J0\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> kf </ci>\n"
           "              <ci> S1 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"_J1\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> kb </ci>\n"
           "              <ci> S2 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string SimpleFlux::modelName() {
    return "SimpleFlux";
}

rr::Matrix<double> SimpleFlux::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,  10,                 1},
                                      {1,  9.062507112353002,  1.9374928876470021},
                                      {2,  8.222669057698957,  2.7773309423010475},
                                      {3,  7.4703138912420455, 3.529686108757959},
                                      {4,  6.796328282346153,  4.203671717653851},
                                      {5,  6.19254890657869,   4.807451093421314},
                                      {6,  5.651662724630082,  5.348337275369921},
                                      {7,  5.167118581013939,  5.832881418986066},
                                      {8,  4.733048693535601,  6.266951306464403},
                                      {9,  4.344191830185,     6.655808169815003},
                                      {10, 3.995839136128227,  7.004160863871776},

                              });
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    10}
    };
}

StringDoubleMap SimpleFlux::steadyState() {
    return StringDoubleMap(
            {
                    {"S1", 1.0},
                    {"S2", 10.0}
            }
    );
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"moiety_conservation", true}
    };
}

rr::Matrix<double> SimpleFlux::fullJacobianAmt() {
    return fullJacobianConc(); // should this be the same?
}

rr::Matrix<double> SimpleFlux::fullJacobianConc() {
    return rr::Matrix<double>(
            {
                    {-0.1, 0.01},
                    {0.1,  -0.01},
            }
    );
}

rr::Matrix<double> SimpleFlux::reducedJacobianAmt() {
    return reducedJacobianConc();
}

rr::Matrix<double> SimpleFlux::reducedJacobianConc() {
    auto mat = rr::Matrix<double>({{-0.11}});
    mat.setRowNames({"S1"});
    mat.setColNames({"S1"});
    return mat;
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::jacobianSettings() {
    return std::unordered_map<std::string, rr::Setting>{{"time", 0.0}};
}

std::vector<std::complex<double>> SimpleFlux::reducedEigenValues() {
    return std::vector<std::complex<double>>(
            {
                    std::complex<double>(-0.11, 0),
            });
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::eigenSettings() {
    return std::unordered_map<std::string, rr::Setting>{{"time", 0}};
}

std::vector<std::complex<double>> SimpleFlux::fullEigenValues() {
    return std::vector<std::complex<double>>(
            {
                    std::complex<double>(-0.11, 0),
                    std::complex<double>(0, 0)
            });
}

rr::Matrix<double> SimpleFlux::linkMatrix() {
    return rr::Matrix<double>(
            {
                    {1},
                    {-1}
            });
}

rr::Matrix<double> SimpleFlux::NrMatrix() {
    return rr::Matrix<double>({{-1, 1}});
}

rr::Matrix<double> SimpleFlux::KMatrix() {
    return rr::Matrix<double>(
            {
                    {1},
                    {1}
            });
}

rr::Matrix<double> SimpleFlux::reducedStoicMatrix() {
    return rr::Matrix<double>({{-1, 1}});
}

rr::Matrix<double> SimpleFlux::fullStoicMatrix() {
    return rr::Matrix<double>(
            {
                    {-1, 1},
                    {1,  -1},
            });
}

rr::Matrix<double> SimpleFlux::extendedStoicMatrix() {
    return fullStoicMatrix(); // not sure when extended should be used
}

rr::Matrix<double> SimpleFlux::L0Matrix() {
    return rr::Matrix<double>({{-1}});
}

rr::Matrix<double> SimpleFlux::conservationMatrix() {
    return rr::Matrix<double>({{1, 1}});
}

rr::Matrix<double> SimpleFlux::unscaledConcentrationControlCoefficientMatrix() {
    return rr::Matrix<double>(
            {
                    {-9.09091, 9.09091},
                    {9.09091,  -9.09091}
            });
}

rr::Matrix<double> SimpleFlux::scaledConcentrationControlCoefficientMatrix() {
    return rr::Matrix<double>(
            {
                    {-0.909091, 0.909091},
                    {0.0909091, -0.0909091}
            });
}

rr::Matrix<double> SimpleFlux::unscaledFluxControlCoefficientMatrix() {
    return rr::Matrix<double>(
            {
                    {0.0909091, 0.909091},
                    {0.0909091, 0.909091}
            });
}

rr::Matrix<double> SimpleFlux::scaledFluxControlCoefficientMatrix() {
    return rr::Matrix<double>(
            {
                    {0.0909091, 0.909091},
                    {0.0909091, 0.909091},
            });
}

//rr::Matrix<double> SimpleFlux::unscaledParameterElasticity() {
//    return rr::Matrix<double>();
//}

rr::Matrix<double> SimpleFlux::unscaledElasticityMatrix() {
    return rr::Matrix<double>(
            {
                    {0.1, 0},
                    {0,   0.01},
            });
}

rr::Matrix<double> SimpleFlux::scaledElasticityMatrix() {
    return rr::Matrix<double>(
            {
                    {1, 0},
                    {0, 1},
            }
    );
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::mcaSettings() {
    return std::unordered_map<std::string, rr::Setting>{{"time", 0}};
}

std::unordered_map<std::string, rr::Setting> SimpleFlux::timeSeriesSensitivityResultSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    3},
    };
}

rr::Matrix3D<double, double> SimpleFlux::timeSeriesSensitivityResult() {
    // copasi generated
    rr::Matrix3D<double, double> mat(
            {0.0, 5.0, 10},
            {
                    // 0.0
                    {
                            {0.0,                 0.0},
                            {0.0,                0.0}
                    },
                    // 5.0
                    {
                            {-29.89114280149252,  29.891142801528048},
                            {12.677378468328724, -12.67737846850636},
                    },
                    // 10.0
                    {
                            {-36.004745327429255, 36.00474532744258},
                            {30.698815036078738, -30.69881503598992}

                    }
            }
    );
    mat.setRowNames({"S1", "S2"});
    mat.setColNames({"kf", "kb"});
    return mat;
}

std::string SimpleFluxManuallyReduced::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"x\" id=\"x\">\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"10\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"S20\" value=\"1\" constant=\"true\"/>\n"
           "      <parameter id=\"Total\" constant=\"false\"/>\n"
           "      <parameter id=\"kf\" value=\"0.1\" constant=\"true\"/>\n"
           "      <parameter id=\"kb\" value=\"0.01\" constant=\"true\"/>\n"
           "      <parameter id=\"S2\" constant=\"false\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfInitialAssignments>\n"
           "      <initialAssignment symbol=\"Total\">\n"
           "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "          <apply>\n"
           "            <plus/>\n"
           "            <ci> S1 </ci>\n"
           "            <ci> S20 </ci>\n"
           "          </apply>\n"
           "        </math>\n"
           "      </initialAssignment>\n"
           "    </listOfInitialAssignments>\n"
           "    <listOfRules>\n"
           "      <assignmentRule variable=\"S2\">\n"
           "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "          <apply>\n"
           "            <minus/>\n"
           "            <ci> Total </ci>\n"
           "            <ci> S1 </ci>\n"
           "          </apply>\n"
           "        </math>\n"
           "      </assignmentRule>\n"
           "    </listOfRules>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"_J0\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <plus/>\n"
           "              <apply>\n"
           "                <times/>\n"
           "                <apply>\n"
           "                  <minus/>\n"
           "                  <ci> kf </ci>\n"
           "                </apply>\n"
           "                <ci> S1 </ci>\n"
           "              </apply>\n"
           "              <apply>\n"
           "                <times/>\n"
           "                <ci> kb </ci>\n"
           "                <apply>\n"
           "                  <minus/>\n"
           "                  <ci> Total </ci>\n"
           "                  <ci> S1 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string SimpleFluxManuallyReduced::modelName() {
    return "SimpleFluxManuallyReduced";
}

StringDoubleMap SimpleFluxManuallyReduced::steadyState() {
    return StringDoubleMap({
                                   {"S1", 1.0}
                           });
}

std::unordered_map<std::string, rr::Setting> SimpleFluxManuallyReduced::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"moiety_conservation", false}
    };
}

std::string OpenLinearFlux::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"x\" id=\"x\">\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"kin\" value=\"1\" constant=\"true\"/>\n"
           "      <parameter id=\"kf\" value=\"0.1\" constant=\"true\"/>\n"
           "      <parameter id=\"kout\" value=\"0.2\" constant=\"true\"/>\n"
           "      <parameter id=\"kb\" value=\"0.01\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"_J0\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <ci> kin </ci>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"_J1\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> kf </ci>\n"
           "              <ci> S1 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"_J2\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> S2 </ci>\n"
           "              <ci> kout </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

StringDoubleMap OpenLinearFlux::steadyState() {
    return StringDoubleMap({
                                   {"S1", 10},
                                   {"S2", 5},
                           });
}

rr::Matrix<double> OpenLinearFlux::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,  0,                  0},
                                      {1,  0.9516258196428962, 0.04527958486459608},
                                      {2,  1.812692469240729,  0.16429269841019906},
                                      {3,  2.5918177931655006, 0.33587597514094913},
                                      {4,  3.296799539619028,  0.5434443622065769},
                                      {5,  3.9346934028419236, 0.7740906110974121},
                                      {6,  4.51188363871537,   1.0178547271489167},
                                      {7,  5.034146961185131,  1.2671318458422862},
                                      {8,  5.506710357479626,  1.5161930250110387},
                                      {9,  5.934303400893279,  1.7607979250221717},
                                      {10, 6.321205586316753,  1.9978820848173928},
                              });
}

std::unordered_map<std::string, rr::Setting> OpenLinearFlux::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    10},
    };
}

std::unordered_map<std::string, rr::Setting> OpenLinearFlux::timeSeriesSensitivityResultSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    3},
    };
}


rr::Matrix3D<double, double> OpenLinearFlux::timeSeriesSensitivityResult() {
    // copasi generated
    rr::Matrix3D<double, double> mat(
            {0.0, 5.0, 10},
            {
                    // 0.0
                    {
                            // S1               S2
                            {0,                 0},
                            {0,                   0},
                            {0,                      0},
                            {0, 0},
                    },
                    // 5.0
                    {
                            // S1               S2
                            {3.934693402848577, 0.7740906159678929}, // kin
                            {-9.018962447266965,  6.460291117027506}, // kf
                            {-1.554312234475219e-11, -1.134615918164661}, // kout
                            {0, 0} // kb (not used)
                    },
                    // 10.0
                    {
                            // S1               S2
                            {6.32120558682292,  1.9978820443633527}, // kin
                            {-26.416083570177662, 13.528667619562285}, // kf
                            {6.843858812999315e-8,   -5.126794156565806}, // kout
                            {0, 0} // kb (not used)

                    }
            }
    );
    mat.setRowNames({"kin", "kf", "kout", "kb"});
    mat.setColNames({"S1", "S2"});
    return mat;
}

std::string OpenLinearFlux::modelName() {
    return "OpenLinearFlux";
}

std::unordered_map<std::string, rr::Setting> OpenLinearFlux::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>{
//            {"allow_presimulation", true},
            {"allow_presimulation",  false},
            {"presimulation_time",   5},
            {"moiety_conservation",  false},
            {"auto_moiety_analysis", false},
    };
}


rr::Matrix<double> OpenLinearFlux::fullJacobianAmt() {
    return rr::Matrix<double>(
            {
                    {-0.1, 0},
                    {0.1,  -0.2},
            });
}

rr::Matrix<double> OpenLinearFlux::fullJacobianConc() {
    return rr::Matrix<double>(
            {
                    {-0.1, 0},
                    {0.1,  -0.2},
            });
}

std::unordered_map<std::string, rr::Setting> OpenLinearFlux::jacobianSettings() {
    return std::unordered_map<std::string, rr::Setting>();
}


std::string Model269::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n"
           "  <model metaid=\"_case00269\" id=\"case00269\" name=\"case00269\" timeUnits=\"time\">\n"
           "    <listOfFunctionDefinitions>\n"
           "      <functionDefinition id=\"calculate\" name=\"calculate\">\n"
           "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "          <lambda>\n"
           "            <bvar>\n"
           "              <ci> x </ci>\n"
           "            </bvar>\n"
           "            <bvar>\n"
           "              <ci> y </ci>\n"
           "            </bvar>\n"
           "            <bvar>\n"
           "              <ci> z </ci>\n"
           "            </bvar>\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <apply>\n"
           "                <factorial/>\n"
           "                <apply>\n"
           "                  <ceiling/>\n"
           "                  <apply>\n"
           "                    <times/>\n"
           "                    <ci> x </ci>\n"
           "                    <ci> y </ci>\n"
           "                  </apply>\n"
           "                </apply>\n"
           "              </apply>\n"
           "              <apply>\n"
           "                <power/>\n"
           "                <ci> z </ci>\n"
           "                <cn type=\"integer\"> -1 </cn>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </lambda>\n"
           "        </math>\n"
           "      </functionDefinition>\n"
           "    </listOfFunctionDefinitions>\n"
           "    <listOfUnitDefinitions>\n"
           "      <unitDefinition id=\"volume\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"litre\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "      <unitDefinition id=\"substance\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"mole\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "      <unitDefinition id=\"time\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"second\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "    </listOfUnitDefinitions>\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"compartment\" name=\"compartment\" spatialDimensions=\"3\" size=\"1\" units=\"volume\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialAmount=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialAmount=\"0\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"p1\" name=\"p1\" value=\"4\" constant=\"true\"/>\n"
           "      <parameter id=\"p2\" name=\"p2\" value=\"25\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"reaction1\" name=\"reaction1\" reversible=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <ci> calculate </ci>\n"
           "              <ci> S1 </ci>\n"
           "              <ci> p1 </ci>\n"
           "              <ci> p2 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string Model269::modelName() {
    return "Model269";
}

rr::Matrix<double> Model269::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,  1,                   0},
                                      {1,  0.5724999588105951,  0.4275000411894048},
                                      {2,  0.444166735551334,   0.5558332644486657},
                                      {3,  0.3641667355513341,  0.6358332644486658},
                                      {4,  0.2841667355513341,  0.7158332644486659},
                                      {5,  0.22708341964881273, 0.7729165803511872},
                                      {6,  0.1870834196488127,  0.8129165803511871},
                                      {7,  0.1470834196488127,  0.8529165803511871},
                                      {8,  0.10708341964881266, 0.892916580351187},
                                      {9,  0.06708341964881266, 0.9329165803511872},
                                      {10, 0.02708341964881266, 0.9729165803511871},
                              });

}

std::unordered_map<std::string, rr::Setting> Model269::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    10}
    };
}

std::unordered_map<std::string, rr::Setting> Model269::timeSeriesSensitivityResultSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    3},
    };
}


rr::Matrix3D<double, double> Model269::timeSeriesSensitivityResult() {
    // copasi generated
    rr::Matrix3D<double, double> mat(
            {0.0, 5.0, 10},
            {
                    // 0.0
                    {
                            {0.0,                  0.0},
                            {0.0,                  0.0}
                    },
                    // 5.0
                    {
                            {-0.10465397234947493, 0.10465397234940554},
                            {0.007987725795975953, -0.007987725795968181},

                    },
                    // 10.0
                    {
                            {-0.10465397234946018, 0.1046539723494333},
                            {0.015979733787983702, -0.015979733787974126},
                    }
            }
    );
    mat.setRowNames({"p1", "p2"});
    mat.setColNames({"S1", "S2"});
    return mat;
}


std::string Model28::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n"
           "  <model metaid=\"_case00028\" id=\"case00028\" name=\"case00028\" timeUnits=\"time\">\n"
           "    <listOfUnitDefinitions>\n"
           "      <unitDefinition id=\"volume\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"litre\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "      <unitDefinition id=\"substance\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"mole\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "      <unitDefinition id=\"time\">\n"
           "        <listOfUnits>\n"
           "          <unit kind=\"second\" exponent=\"1\" scale=\"0\" multiplier=\"1\"/>\n"
           "        </listOfUnits>\n"
           "      </unitDefinition>\n"
           "    </listOfUnitDefinitions>\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"compartment\" name=\"compartment\" spatialDimensions=\"3\" size=\"1\" units=\"volume\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialAmount=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialAmount=\"0\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"p1\" name=\"p1\" value=\"4\" constant=\"true\"/>\n"
           "      <parameter id=\"p2\" name=\"p2\" value=\"25\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"reaction1\" name=\"reaction1\" reversible=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <apply>\n"
           "                <factorial/>\n"
           "                <apply>\n"
           "                  <ceiling/>\n"
           "                  <apply>\n"
           "                    <times/>\n"
           "                    <ci> p1 </ci>\n"
           "                    <ci> S1 </ci>\n"
           "                  </apply>\n"
           "                </apply>\n"
           "              </apply>\n"
           "              <apply>\n"
           "                <power/>\n"
           "                <ci> p2 </ci>\n"
           "                <cn type=\"integer\"> -1 </cn>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string Model28::modelName() {
    return "Model28";
}

rr::Matrix<double> Model28::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,  1,                   0},
                                      {1,  0.5724999588105951,  0.4275000411894048},
                                      {2,  0.444166735551334,   0.5558332644486657},
                                      {3,  0.3641667355513341,  0.6358332644486658},
                                      {4,  0.2841667355513341,  0.7158332644486659},
                                      {5,  0.22708341964881273, 0.7729165803511872},
                                      {6,  0.1870834196488127,  0.8129165803511871},
                                      {7,  0.1470834196488127,  0.8529165803511871},
                                      {8,  0.10708341964881266, 0.892916580351187},
                                      {9,  0.06708341964881266, 0.9329165803511872},
                                      {10, 0.02708341964881266, 0.9729165803511871},
                              });

}

std::unordered_map<std::string, rr::Setting> Model28::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    10}
    };
}

std::unordered_map<std::string, rr::Setting> Model28::timeSeriesSensitivityResultSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 10},
            {"steps",    3},
    };
}


rr::Matrix3D<double, double> Model28::timeSeriesSensitivityResult() {
    // copasi generated
    rr::Matrix3D<double, double> mat(
            {0.0, 5.0, 10},
            {
                    // 0.0
                    {
                            {0.0,                  0.0},
                            {0.0,                  0.0}
                    },
                    // 5.0
                    {
                            {-0.10465397234947493, 0.10465397234940554},
                            {0.007987725795975953, -0.007987725795968181}

                    },
                    // 10.0
                    {
                            {-0.10465397234946018, 0.1046539723494333},
                            {0.015979733787983702, -0.015979733787974126},
                    }
            }
    );
    mat.setRowNames({"p1", "p2"});
    mat.setColNames({"S1", "S2"});
    return mat;
}


std::string CeilInRateLaw::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"CeilModel1\" id=\"CeilModel1\">\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"10\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"k1\" value=\"104\" constant=\"true\"/>\n"
           "      <parameter id=\"k2\" value=\"39\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"R1\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> S1 </ci>\n"
           "              <apply>\n"
           "                <ceiling/>\n"
           "                <apply>\n"
           "                  <divide/>\n"
           "                  <ci> k1 </ci>\n"
           "                  <ci> k2 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string CeilInRateLaw::modelName() {
    return "CeilInRateLaw";
}

rr::Matrix<double> CeilInRateLaw::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,   10,                    0},
                                      {0.4, 3.0119408731651975,    6.9880591268348},
                                      {0.8, 0.9071782094955865,    9.09282179050441},
                                      {1.2, 0.2732365534742893,    9.726763446525707},
                                      {1.6, 0.08229718872631311,   9.917702811273687},
                                      {2,   0.024787413004265253,  9.975212586995736},
                                      {2.4, 0.007465818043742018,  9.992534181956255},
                                      {2.8, 0.002248658948996144,  9.997751341051005},
                                      {3.2, 0.0006772823748483675, 9.999322717625152},
                                      {3.6, 0.0002039933214930827, 9.999796006678507},
                                      {4,   6.144154589603753e-5,  9.999938558454103},
                              });

}

std::unordered_map<std::string, rr::Setting> CeilInRateLaw::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 4},
            {"steps",    10}
    };
}


std::string FactorialInRateLaw::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"FactorialModel1\" id=\"FactorialModel1\">\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"default_compartment\" initialConcentration=\"10\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" compartment=\"default_compartment\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"k1\" value=\"104\" constant=\"true\"/>\n"
           "      <parameter id=\"k2\" value=\"39\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"R1\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> S1 </ci>\n"
           "              <apply>\n"
           "                <factorial/>\n"
           "                <apply>\n"
           "                  <divide/>\n"
           "                  <ci> k1 </ci>\n"
           "                  <ci> k2 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string FactorialInRateLaw::modelName() {
    return "FactorialInRateLaw";
}

rr::Matrix<double> FactorialInRateLaw::timeSeriesResult() {
    return rr::Matrix<double>({
                                      {0,   10,        0},
                                      {0.2, 5.52892,   4.47108},
                                      {0.4, 3.0569,    6.9431},
                                      {0.6, 1.69013,   8.30987},
                                      {0.8, 0.93446,   9.06554},
                                      {1,   0.516655,  9.48334},
                                      {1.2, 0.285655,  9.71435},
                                      {1.4, 0.157936,  9.84206},
                                      {1.6, 0.0873215, 9.91268},
                                      {1.8, 0.0482793, 9.95172},
                                      {2,   0.0266933, 9.97331},
                              });

}

std::unordered_map<std::string, rr::Setting> FactorialInRateLaw::timeSeriesSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"start",    0},
            {"duration", 2},
            {"steps",    10}
    };
}


std::string Venkatraman2010::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"__main\" id=\"__main\">\n"
           "    <listOfCompartments>\n"
           "      <compartment sboTerm=\"SBO:0000410\" id=\"default_compartment\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"scUPA\" compartment=\"default_compartment\" initialConcentration=\"1.16213e-8\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"PLG\" compartment=\"default_compartment\" initialConcentration=\"0.0262792\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"PLS\" compartment=\"default_compartment\" initialConcentration=\"19.7847\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"tcUPA\" compartment=\"default_compartment\" initialConcentration=\"19.9809\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"keff1\" value=\"0.0017\" constant=\"true\"/>\n"
           "      <parameter id=\"keff2\" value=\"1\" constant=\"true\"/>\n"
           "      <parameter id=\"keff3\" value=\"0.03\" constant=\"true\"/>\n"
           "      <parameter id=\"n\" value=\"3\" constant=\"true\"/>\n"
           "      <parameter id=\"u1\" value=\"0.0001\" constant=\"true\"/>\n"
           "      <parameter id=\"u3\" value=\"0.0001\" constant=\"true\"/>\n"
           "      <parameter id=\"u2\" value=\"0.001\" constant=\"true\"/>\n"
           "      <parameter id=\"u4\" value=\"0.001\" constant=\"true\"/>\n"
           "      <parameter id=\"alpha1\" value=\"9e-05\" constant=\"true\"/>\n"
           "      <parameter id=\"alpha2\" value=\"0.001\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"r1\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"scUPA\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <ci> alpha1 </ci>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r2\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"scUPA\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> scUPA </ci>\n"
           "              <ci> u1 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r3\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"PLG\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <ci> alpha2 </ci>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r4\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PLG\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> PLG </ci>\n"
           "              <ci> u2 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r5\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PLG\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"PLS\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"scUPA\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> keff1 </ci>\n"
           "              <ci> scUPA </ci>\n"
           "              <ci> PLG </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r6\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"scUPA\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"tcUPA\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"PLS\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> keff2 </ci>\n"
           "              <ci> scUPA </ci>\n"
           "              <apply>\n"
           "                <power/>\n"
           "                <ci> PLS </ci>\n"
           "                <ci> n </ci>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r7\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PLG\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"PLS\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"tcUPA\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> keff3 </ci>\n"
           "              <ci> tcUPA </ci>\n"
           "              <ci> PLG </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r8\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"tcUPA\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> tcUPA </ci>\n"
           "              <ci> u3 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"r9\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PLS\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> PLS </ci>\n"
           "              <ci> u4 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string Venkatraman2010::modelName() {
    return "Venkatraman2010";
}

StringDoubleMap Venkatraman2010::steadyState() {
    return StringDoubleMap{
            {"scUPA", 0.00010036488071501325},
            {"PLG",   0.03571790894678159},
            {"PLS",   0.9642820910532185},
            {"tcUPA", 0.8998996351192852}
    };
}

std::unordered_map<std::string, rr::Setting> Venkatraman2010::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>(
            {
                    {"allow_presimulation", true},
                    {"presimulation_time",  100000},
                    {"moiety_conservation", false},

            });
}

rr::Matrix<double> Venkatraman2010::fullJacobianAmt() {
    return fullJacobianConc();
}

rr::Matrix<double> Venkatraman2010::fullJacobianConc() {
    return rr::Matrix<double>({
                                      {-7744.4113653644172, 0,         -1.36469e-05, 0},
                                      {-4.46746e-05,        -0.600427, 0,            -0.000788376},
                                      {4.46746e-05,         0.599427,  -0.001,       0.000788376},
                                      {7744.4113653644172,  0,         1.36469e-05,  -0.0001}
                              });
}

std::unordered_map<std::string, rr::Setting> Venkatraman2010::jacobianSettings() {
    return std::unordered_map<std::string, rr::Setting>{{"time", 0.0}};
}


std::string Brown2004::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
           "  <model metaid=\"Brown2004_NGF_EGF_signaling\" id=\"Brown2004_NGF_EGF_signaling\">\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"cell\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"EGF\" compartment=\"cell\" initialConcentration=\"10002000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"NGF\" compartment=\"cell\" initialConcentration=\"456000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"freeEGFReceptor\" compartment=\"cell\" initialConcentration=\"80000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"boundEGFReceptor\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"freeNGFReceptor\" compartment=\"cell\" initialConcentration=\"10000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"boundNGFReceptor\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"SosInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"SosActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"P90RskInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"P90RskActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"RasInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"RasActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"RasGapActive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
           "      <species id=\"Raf1Inactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"Raf1Active\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"BRafInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"BRafActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"MekInactive\" compartment=\"cell\" initialConcentration=\"600000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"MekActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"ErkInactive\" compartment=\"cell\" initialConcentration=\"600000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"ErkActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"PI3KInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"PI3KActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"AktInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"AktActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"C3GInactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"C3GActive\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"Rap1Inactive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"Rap1Active\" compartment=\"cell\" initialConcentration=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"RapGapActive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
           "      <species id=\"PP2AActive\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
           "      <species id=\"Raf1PPtase\" compartment=\"cell\" initialConcentration=\"120000\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"true\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"krbEGF\" value=\"2.18503e-05\" constant=\"true\"/>\n"
           "      <parameter id=\"kruEGF\" value=\"0.0121008\" constant=\"true\"/>\n"
           "      <parameter id=\"krbNGF\" value=\"1.38209e-07\" constant=\"true\"/>\n"
           "      <parameter id=\"kruNGF\" value=\"0.00723811\" constant=\"true\"/>\n"
           "      <parameter id=\"kEGF\" value=\"694.731\" constant=\"true\"/>\n"
           "      <parameter id=\"KmEGF\" value=\"6086070\" constant=\"true\"/>\n"
           "      <parameter id=\"kNGF\" value=\"389.428\" constant=\"true\"/>\n"
           "      <parameter id=\"KmNGF\" value=\"2112.66\" constant=\"true\"/>\n"
           "      <parameter id=\"kdSos\" value=\"1611.97\" constant=\"true\"/>\n"
           "      <parameter id=\"KmdSos\" value=\"896896\" constant=\"true\"/>\n"
           "      <parameter id=\"kSos\" value=\"32.344\" constant=\"true\"/>\n"
           "      <parameter id=\"KmSos\" value=\"35954.3\" constant=\"true\"/>\n"
           "      <parameter id=\"kRasGap\" value=\"1509.36\" constant=\"true\"/>\n"
           "      <parameter id=\"KmRasGap\" value=\"1432410\" constant=\"true\"/>\n"
           "      <parameter id=\"kRasToRaf1\" value=\"0.884096\" constant=\"true\"/>\n"
           "      <parameter id=\"KmRasToRaf1\" value=\"62464.6\" constant=\"true\"/>\n"
           "      <parameter id=\"kpRaf1\" value=\"185.759\" constant=\"true\"/>\n"
           "      <parameter id=\"KmpRaf1\" value=\"4768350\" constant=\"true\"/>\n"
           "      <parameter id=\"kpBRaf\" value=\"125.089\" constant=\"true\"/>\n"
           "      <parameter id=\"KmpBRaf\" value=\"157948\" constant=\"true\"/>\n"
           "      <parameter id=\"kpMekCytoplasmic\" value=\"9.85367\" constant=\"true\"/>\n"
           "      <parameter id=\"KmpMekCytoplasmic\" value=\"1007340\" constant=\"true\"/>\n"
           "      <parameter id=\"kdMek\" value=\"2.83243\" constant=\"true\"/>\n"
           "      <parameter id=\"KmdMek\" value=\"518753\" constant=\"true\"/>\n"
           "      <parameter id=\"kdErk\" value=\"8.8912\" constant=\"true\"/>\n"
           "      <parameter id=\"KmdErk\" value=\"3496490\" constant=\"true\"/>\n"
           "      <parameter id=\"kdRaf1\" value=\"0.126329\" constant=\"true\"/>\n"
           "      <parameter id=\"KmdRaf1\" value=\"1061.71\" constant=\"true\"/>\n"
           "      <parameter id=\"kpP90Rsk\" value=\"0.0213697\" constant=\"true\"/>\n"
           "      <parameter id=\"KmpP90Rsk\" value=\"763523\" constant=\"true\"/>\n"
           "      <parameter id=\"kPI3K\" value=\"10.6737\" constant=\"true\"/>\n"
           "      <parameter id=\"KmPI3K\" value=\"184912\" constant=\"true\"/>\n"
           "      <parameter id=\"kPI3KRas\" value=\"0.0771067\" constant=\"true\"/>\n"
           "      <parameter id=\"KmPI3KRas\" value=\"272056\" constant=\"true\"/>\n"
           "      <parameter id=\"kAkt\" value=\"0.0566279\" constant=\"true\"/>\n"
           "      <parameter id=\"KmAkt\" value=\"653951\" constant=\"true\"/>\n"
           "      <parameter id=\"kdRaf1ByAkt\" value=\"15.1212\" constant=\"true\"/>\n"
           "      <parameter id=\"KmRaf1ByAkt\" value=\"119355\" constant=\"true\"/>\n"
           "      <parameter id=\"kC3GNGF\" value=\"146.912\" constant=\"true\"/>\n"
           "      <parameter id=\"KmC3GNGF\" value=\"12876.2\" constant=\"true\"/>\n"
           "      <parameter id=\"kC3G\" value=\"1.40145\" constant=\"true\"/>\n"
           "      <parameter id=\"KmC3G\" value=\"10965.6\" constant=\"true\"/>\n"
           "      <parameter id=\"kRapGap\" value=\"27.265\" constant=\"true\"/>\n"
           "      <parameter id=\"KmRapGap\" value=\"295990\" constant=\"true\"/>\n"
           "      <parameter id=\"kRap1ToBRaf\" value=\"2.20995\" constant=\"true\"/>\n"
           "      <parameter id=\"KmRap1ToBRaf\" value=\"1025460\" constant=\"true\"/>\n"
           "      <parameter id=\"kdBRaf\" value=\"441.287\" constant=\"true\"/>\n"
           "      <parameter id=\"KmdBRaf\" value=\"10879500\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"EGFBindingReaction\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"EGF\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "          <speciesReference species=\"freeEGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"boundEGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <ci> krbEGF </ci>\n"
           "              <ci> EGF </ci>\n"
           "              <ci> freeEGFReceptor </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"EGFUnbindingReaction\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"boundEGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"EGF\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "          <speciesReference species=\"freeEGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <ci> kruEGF </ci>\n"
           "              <ci> boundEGFReceptor </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"NGFBindingReaction\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"NGF\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "          <speciesReference species=\"freeNGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"boundNGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> krbNGF </ci>\n"
           "              <ci> NGF </ci>\n"
           "              <ci> freeNGFReceptor </ci>\n"
           "              <ci> cell </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"NGFUnbindingReaction\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"boundNGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"NGF\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "          <speciesReference species=\"freeNGFReceptor\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> kruNGF </ci>\n"
           "              <ci> boundNGFReceptor </ci>\n"
           "              <ci> cell </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"SosActivationByEGFReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"SosInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"SosActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"boundEGFReceptor\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kEGF </ci>\n"
           "                  <ci> boundEGFReceptor </ci>\n"
           "                  <ci> SosInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> SosInactive </ci>\n"
           "                  <ci> KmEGF </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"SosActivationByNGFReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"SosInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"SosActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"boundNGFReceptor\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kNGF </ci>\n"
           "                  <ci> boundNGFReceptor </ci>\n"
           "                  <ci> SosInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> SosInactive </ci>\n"
           "                  <ci> KmNGF </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"SosDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"SosActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"SosInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"P90RskActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdSos </ci>\n"
           "                  <ci> P90RskActive </ci>\n"
           "                  <ci> SosActive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> SosActive </ci>\n"
           "                  <ci> KmdSos </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"RasActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"RasInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"RasActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"SosActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kSos </ci>\n"
           "                  <ci> SosActive </ci>\n"
           "                  <ci> RasInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> RasInactive </ci>\n"
           "                  <ci> KmSos </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"RasDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"RasActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"RasInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"RasGapActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kRasGap </ci>\n"
           "                  <ci> RasGapActive </ci>\n"
           "                  <ci> RasActive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> RasActive </ci>\n"
           "                  <ci> KmRasGap </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Raf1ByRasActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"Raf1Inactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"Raf1Active\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"RasActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kRasToRaf1 </ci>\n"
           "                  <ci> RasActive </ci>\n"
           "                  <ci> Raf1Inactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> Raf1Inactive </ci>\n"
           "                  <ci> KmRasToRaf1 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"MekbyRaf1ActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"MekInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"MekActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"Raf1Active\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kpRaf1 </ci>\n"
           "                  <ci> Raf1Active </ci>\n"
           "                  <ci> MekInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> MekInactive </ci>\n"
           "                  <ci> KmpRaf1 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"MekbyBRafActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"MekInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"MekActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"BRafActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kpBRaf </ci>\n"
           "                  <ci> BRafActive </ci>\n"
           "                  <ci> MekInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> MekInactive </ci>\n"
           "                  <ci> KmpBRaf </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"ErkActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"ErkInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"ErkActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"MekActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kpMekCytoplasmic </ci>\n"
           "                  <ci> MekActive </ci>\n"
           "                  <ci> ErkInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> ErkInactive </ci>\n"
           "                  <ci> KmpMekCytoplasmic </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"MekDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"MekActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"MekInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"PP2AActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdMek </ci>\n"
           "                  <ci> PP2AActive </ci>\n"
           "                  <ci> MekActive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> MekActive </ci>\n"
           "                  <ci> KmdMek </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"ErkDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"ErkActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"ErkInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"PP2AActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdErk </ci>\n"
           "                  <ci> PP2AActive </ci>\n"
           "                  <ci> ErkActive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> ErkActive </ci>\n"
           "                  <ci> KmdErk </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Raf1byPPtaseDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"Raf1Active\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"Raf1Inactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"Raf1PPtase\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdRaf1 </ci>\n"
           "                  <ci> Raf1PPtase </ci>\n"
           "                  <ci> Raf1Active </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> Raf1Active </ci>\n"
           "                  <ci> KmdRaf1 </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"P90RskActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"P90RskInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"P90RskActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"ErkActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kpP90Rsk </ci>\n"
           "                  <ci> ErkActive </ci>\n"
           "                  <ci> P90RskInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> P90RskInactive </ci>\n"
           "                  <ci> KmpP90Rsk </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"PI3KbyEGFRActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PI3KInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"PI3KActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"boundEGFReceptor\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kPI3K </ci>\n"
           "                  <ci> boundEGFReceptor </ci>\n"
           "                  <ci> PI3KInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> PI3KInactive </ci>\n"
           "                  <ci> KmPI3K </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"PI3KbyRasActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"PI3KInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"PI3KActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"RasActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kPI3KRas </ci>\n"
           "                  <ci> RasActive </ci>\n"
           "                  <ci> PI3KInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> PI3KInactive </ci>\n"
           "                  <ci> KmPI3KRas </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"AktActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"AktInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"AktActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"PI3KActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kAkt </ci>\n"
           "                  <ci> PI3KActive </ci>\n"
           "                  <ci> AktInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> AktInactive </ci>\n"
           "                  <ci> KmAkt </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Raf1ByAktDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"Raf1Active\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"Raf1Inactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"AktActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdRaf1ByAkt </ci>\n"
           "                  <ci> AktActive </ci>\n"
           "                  <ci> Raf1Active </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> Raf1Active </ci>\n"
           "                  <ci> KmRaf1ByAkt </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"C3GActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"C3GInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"C3GActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"boundNGFReceptor\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kC3GNGF </ci>\n"
           "                  <ci> boundNGFReceptor </ci>\n"
           "                  <ci> C3GInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> C3GInactive </ci>\n"
           "                  <ci> KmC3GNGF </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Rap1ActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"Rap1Inactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"Rap1Active\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"C3GActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kC3G </ci>\n"
           "                  <ci> C3GActive </ci>\n"
           "                  <ci> Rap1Inactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> Rap1Inactive </ci>\n"
           "                  <ci> KmC3G </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Rap1DeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"Rap1Active\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"Rap1Inactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"RapGapActive\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kRapGap </ci>\n"
           "                  <ci> RapGapActive </ci>\n"
           "                  <ci> Rap1Active </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> Rap1Active </ci>\n"
           "                  <ci> KmRapGap </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"BRafByRap1ActivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"BRafInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"BRafActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"Rap1Active\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kRap1ToBRaf </ci>\n"
           "                  <ci> Rap1Active </ci>\n"
           "                  <ci> BRafInactive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> BRafInactive </ci>\n"
           "                  <ci> KmRap1ToBRaf </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"BRafbyPPtaseDeactivationReaction\" reversible=\"true\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"BRafActive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"BRafInactive\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <listOfModifiers>\n"
           "          <modifierSpeciesReference species=\"Raf1PPtase\"/>\n"
           "        </listOfModifiers>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> cell </ci>\n"
           "              <apply>\n"
           "                <divide/>\n"
           "                <apply>\n"
           "                  <times/>\n"
           "                  <ci> kdBRaf </ci>\n"
           "                  <ci> Raf1PPtase </ci>\n"
           "                  <ci> BRafActive </ci>\n"
           "                </apply>\n"
           "                <apply>\n"
           "                  <plus/>\n"
           "                  <ci> BRafActive </ci>\n"
           "                  <ci> KmdBRaf </ci>\n"
           "                </apply>\n"
           "              </apply>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string Brown2004::modelName() {
    return "Brown2004";
}

/**
 * Copasi output:
 *
 *
 */
StringDoubleMap Brown2004::steadyState() {
    return StringDoubleMap{
            {"EGF",              9922004.46501588},
            {"NGF",              447048.6328396933},
            {"freeEGFReceptor",  4.465015881563886},
            {"boundEGFReceptor", 79995.53498411844},
            {"freeNGFReceptor",  1048.632839693278},
            {"boundNGFReceptor", 8951.367160306723},
            {"SosInactive",      99569.9548813399},
            {"SosActive",        20430.0451186601},
            {"P90RskInactive",   0},
            {"P90RskActive",     120000},
            {"RasInactive",      115999.51997363668},
            {"RasActive",        4000.4800263633224},
            {"Raf1Inactive",     119918.26132477315},
            {"Raf1Active",       81.73867522685032},
            {"BRafInactive",     119315.8938154474},
            {"BRafActive",       684.1061845525983},
            {"MekInactive",      475871.3129791245},
            {"MekActive",        124128.68702087551},
            {"ErkInactive",      118834.97021061042},
            {"ErkActive",        481165.02978938946},
            {"PI3KInactive",     0},
            {"PI3KActive",       119999.99999999997},
            {"AktInactive",      0},
            {"AktActive",        120000},
            {"C3GInactive",      0},
            {"C3GActive",        120000},
            {"Rap1Inactive",     105544.62439152892},
            {"Rap1Active",       14455.37560847107},
    };

};


std::unordered_map<std::string, rr::Setting> Brown2004::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"allow_presimulation", true},
            // this model can converge to a set of values different to the reference data,
            // depending on starting values. For this reason, start the list of presimulation
            // times at a larger number.
            // Note: the reference data is from copasi, which presimulates to 1000.
            // We can actually get a steady state from an earlier presimulation point,
            // but since the reference data is from copasi, we stick to using 1000.
            {"presimulation_times", std::vector<double>({1000, 1e4})},
    };
}

//rr::Matrix<double> Brown2004::timeSeriesResult() {
//    return rr::Matrix<double>({
//                                    // EGF      	NGF     	freeEGFReceptor 	boundEGFReceptor	freeNGFReceptor	boundNGFReceptor	SosInactive	SosActive	P90RskInactive	P90RskActive	RasInactive	RasActive	Raf1Inactive	Raf1Active	BRafInactive	BRafActive	MekInactive	MekActive	ErkInactive	ErkActive	PI3KInactive	PI3KActive	AktInactive	AktActive	C3GInactive	C3GActive	Rap1Inactive	Rap1Active
//                                      {10002000,	456000,	80000,	0,	10000,	0,	120000,	0,	120000,	0,	120000,	0,	120000,	0,	120000,	0,	600000,	0,	600000,	0,	120000,	0,	120000,	0,	120000,	0,	120000,	0},
//                                      {9922004.465, 453349.8606, 4.465015882, 79995.53498, 7349.860605, 2650.139395, 4259.574671, 115740.4253, 116032.427,  3967.573017, 98004.70168, 21995.29832, 116937.5552, 3062.444807, 119315.8939, 684.1061226, 395680.9755, 204319.0245, 81268.04632, 518731.9537, 2.14E-05, 120000, 115124.6722, 4875.327838, 9.89E-21, 120000, 105544.6244, 14455.37561},
//                                      {9922004.465, 451491.1631, 4.465015882, 79995.53498, 5491.163092, 4508.836908, 20935.13104, 99064.86896, 108842.6175, 11157.38249, 101057.3069, 18942.69307, 118385.5037, 1614.496273, 119315.8938, 684.1061846, 408801.6501, 191198.3499, 78498.1564,  521501.8436, 7.10E-14, 120000, 110133.3699, 9866.630116, 1.92E-27, 120000, 105544.6244, 14455.37561}
//                              });
//}
//
//std::unordered_map<std::string, rr::Setting> Brown2004::timeSeriesSettings() {
//    return std::unordered_map<std::string, rr::Setting>({
//                                                                {"start",    0},
//                                                                {"duration", 10},
//                                                                {"steps",    2}
//                                                        });
//}


std::string LayoutOnly::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:layout=\"http://www.sbml.org/sbml/level3/version1/layout/version1\" level=\"3\" version=\"1\" layout:required=\"false\">\n"
           "  <model id=\"TestModel\">\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"Compartment_1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"Species_1\" compartment=\"Compartment_1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"Species_2\" compartment=\"Compartment_1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"Reaction_1\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference id=\"SpeciesReference_1\" species=\"Species_1\" constant=\"false\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference id=\"SpeciesReference_2\" species=\"Species_2\" constant=\"false\"/>\n"
           "        </listOfProducts>\n"
           "      </reaction>\n"
           "      <reaction id=\"Reaction_2\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference id=\"SpeciesReference_3\" species=\"Species_2\" constant=\"false\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference id=\"SpeciesReference_4\" species=\"Species_1\" constant=\"false\"/>\n"
           "        </listOfProducts>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "    <layout:listOfLayouts xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:layout=\"http://www.sbml.org/sbml/level3/version1/layout/version1\">\n"
           "      <layout:layout layout:id=\"Layout_1\">\n"
           "        <layout:dimensions layout:width=\"400\" layout:height=\"220\"/>\n"
           "        <layout:listOfCompartmentGlyphs>\n"
           "          <layout:compartmentGlyph layout:id=\"CompartmentGlyph_1\" layout:compartment=\"Compartment_1\">\n"
           "            <layout:boundingBox layout:id=\"bb1\">\n"
           "              <layout:position layout:x=\"5\" layout:y=\"5\"/>\n"
           "              <layout:dimensions layout:width=\"390\" layout:height=\"210\"/>\n"
           "            </layout:boundingBox>\n"
           "          </layout:compartmentGlyph>\n"
           "        </layout:listOfCompartmentGlyphs>\n"
           "        <layout:listOfSpeciesGlyphs>\n"
           "          <layout:speciesGlyph layout:id=\"SpeciesGlyph_1\" layout:species=\"Species_1\">\n"
           "            <layout:boundingBox layout:id=\"bb2\">\n"
           "              <layout:position layout:x=\"80\" layout:y=\"26\"/>\n"
           "              <layout:dimensions layout:width=\"240\" layout:height=\"24\"/>\n"
           "            </layout:boundingBox>\n"
           "          </layout:speciesGlyph>\n"
           "          <layout:speciesGlyph layout:id=\"SpeciesGlyph_2\" layout:species=\"Species_2\">\n"
           "            <layout:boundingBox layout:id=\"bb3\">\n"
           "              <layout:position layout:x=\"80\" layout:y=\"170\"/>\n"
           "              <layout:dimensions layout:width=\"240\" layout:height=\"24\"/>\n"
           "            </layout:boundingBox>\n"
           "          </layout:speciesGlyph>\n"
           "        </layout:listOfSpeciesGlyphs>\n"
           "        <layout:listOfReactionGlyphs>\n"
           "          <layout:reactionGlyph layout:id=\"ReactionGlyph_1\" layout:reaction=\"Reaction_1\">\n"
           "            <layout:curve>\n"
           "              <layout:listOfCurveSegments>\n"
           "                <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"LineSegment\">\n"
           "                  <layout:start layout:x=\"165\" layout:y=\"105\"/>\n"
           "                  <layout:end layout:x=\"165\" layout:y=\"115\"/>\n"
           "                </layout:curveSegment>\n"
           "              </layout:listOfCurveSegments>\n"
           "            </layout:curve>\n"
           "            <layout:listOfSpeciesReferenceGlyphs>\n"
           "              <layout:speciesReferenceGlyph layout:id=\"SpeciesReferenceGlyph_1\" layout:speciesReference=\"SpeciesReference_1\" layout:speciesGlyph=\"SpeciesGlyph_1\" layout:role=\"substrate\">\n"
           "                <layout:curve>\n"
           "                  <layout:listOfCurveSegments>\n"
           "                    <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"CubicBezier\">\n"
           "                      <layout:start layout:x=\"165\" layout:y=\"105\"/>\n"
           "                      <layout:end layout:x=\"195\" layout:y=\"60\"/>\n"
           "                      <layout:basePoint1 layout:x=\"165\" layout:y=\"90\"/>\n"
           "                      <layout:basePoint2 layout:x=\"165\" layout:y=\"90\"/>\n"
           "                    </layout:curveSegment>\n"
           "                  </layout:listOfCurveSegments>\n"
           "                </layout:curve>\n"
           "              </layout:speciesReferenceGlyph>\n"
           "              <layout:speciesReferenceGlyph layout:id=\"SpeciesReferenceGlyph_2\" layout:speciesReference=\"SpeciesReference_2\" layout:speciesGlyph=\"SpeciesGlyph_2\" layout:role=\"product\">\n"
           "                <layout:curve>\n"
           "                  <layout:listOfCurveSegments>\n"
           "                    <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"CubicBezier\">\n"
           "                      <layout:start layout:x=\"165\" layout:y=\"115\"/>\n"
           "                      <layout:end layout:x=\"195\" layout:y=\"160\"/>\n"
           "                      <layout:basePoint1 layout:x=\"165\" layout:y=\"130\"/>\n"
           "                      <layout:basePoint2 layout:x=\"165\" layout:y=\"130\"/>\n"
           "                    </layout:curveSegment>\n"
           "                  </layout:listOfCurveSegments>\n"
           "                </layout:curve>\n"
           "              </layout:speciesReferenceGlyph>\n"
           "            </layout:listOfSpeciesReferenceGlyphs>\n"
           "          </layout:reactionGlyph>\n"
           "          <layout:reactionGlyph layout:id=\"ReactionGlyph_2\" layout:reaction=\"Reaction_2\">\n"
           "            <layout:curve>\n"
           "              <layout:listOfCurveSegments>\n"
           "                <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"LineSegment\">\n"
           "                  <layout:start layout:x=\"235\" layout:y=\"105\"/>\n"
           "                  <layout:end layout:x=\"235\" layout:y=\"115\"/>\n"
           "                </layout:curveSegment>\n"
           "              </layout:listOfCurveSegments>\n"
           "            </layout:curve>\n"
           "            <layout:listOfSpeciesReferenceGlyphs>\n"
           "              <layout:speciesReferenceGlyph layout:id=\"SpeciesReferenceGlyph_3\" layout:speciesReference=\"SpeciesReference_3\" layout:speciesGlyph=\"SpeciesGlyph_2\" layout:role=\"substrate\">\n"
           "                <layout:curve>\n"
           "                  <layout:listOfCurveSegments>\n"
           "                    <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"CubicBezier\">\n"
           "                      <layout:start layout:x=\"235\" layout:y=\"115\"/>\n"
           "                      <layout:end layout:x=\"205\" layout:y=\"160\"/>\n"
           "                      <layout:basePoint1 layout:x=\"235\" layout:y=\"130\"/>\n"
           "                      <layout:basePoint2 layout:x=\"235\" layout:y=\"130\"/>\n"
           "                    </layout:curveSegment>\n"
           "                  </layout:listOfCurveSegments>\n"
           "                </layout:curve>\n"
           "              </layout:speciesReferenceGlyph>\n"
           "              <layout:speciesReferenceGlyph layout:id=\"SpeciesReferenceGlyph_4\" layout:speciesReference=\"SpeciesReference_4\" layout:speciesGlyph=\"SpeciesGlyph_1\" layout:role=\"product\">\n"
           "                <layout:curve>\n"
           "                  <layout:listOfCurveSegments>\n"
           "                    <layout:curveSegment xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:type=\"CubicBezier\">\n"
           "                      <layout:start layout:x=\"235\" layout:y=\"105\"/>\n"
           "                      <layout:end layout:x=\"205\" layout:y=\"60\"/>\n"
           "                      <layout:basePoint1 layout:x=\"235\" layout:y=\"90\"/>\n"
           "                      <layout:basePoint2 layout:x=\"235\" layout:y=\"90\"/>\n"
           "                    </layout:curveSegment>\n"
           "                  </layout:listOfCurveSegments>\n"
           "                </layout:curve>\n"
           "              </layout:speciesReferenceGlyph>\n"
           "            </layout:listOfSpeciesReferenceGlyphs>\n"
           "          </layout:reactionGlyph>\n"
           "        </layout:listOfReactionGlyphs>\n"
           "        <layout:listOfTextGlyphs>\n"
           "          <layout:textGlyph layout:id=\"TextGlyph_01\" layout:originOfText=\"SpeciesGlyph_1\" layout:graphicalObject=\"SpeciesGlyph_1\">\n"
           "            <layout:boundingBox layout:id=\"bbA\">\n"
           "              <layout:position layout:x=\"92\" layout:y=\"26\"/>\n"
           "              <layout:dimensions layout:width=\"228\" layout:height=\"24\"/>\n"
           "            </layout:boundingBox>\n"
           "          </layout:textGlyph>\n"
           "          <layout:textGlyph layout:id=\"TextGlyph_02\" layout:originOfText=\"SpeciesGlyph_2\" layout:graphicalObject=\"SpeciesGlyph_2\">\n"
           "            <layout:boundingBox layout:id=\"bbB\">\n"
           "              <layout:position layout:x=\"92\" layout:y=\"170\"/>\n"
           "              <layout:dimensions layout:width=\"228\" layout:height=\"24\"/>\n"
           "            </layout:boundingBox>\n"
           "          </layout:textGlyph>\n"
           "        </layout:listOfTextGlyphs>\n"
           "      </layout:layout>\n"
           "    </layout:listOfLayouts>\n"
           "  </model>\n"
           "</sbml>";
}

std::string LayoutOnly::modelName() {
    return "LayoutOnly";
}

std::string ModelWithLocalParameters::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:layout=\"http://www.sbml.org/sbml/level3/version1/layout/version1\" xmlns:render=\"http://www.sbml.org/sbml/level3/version1/render/version1\" level=\"3\" version=\"1\" layout:required=\"false\" render:required=\"false\">\n"
           "  <model id=\"ModelWithLocalParameters\" name=\"ModelWithLocalParameters\" substanceUnits=\"substance\" timeUnits=\"time\" volumeUnits=\"volume\" areaUnits=\"area\" lengthUnits=\"length\" extentUnits=\"substance\">\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"compartment\" name=\"compartment\" spatialDimensions=\"3\" size=\"1\" units=\"volume\" constant=\"true\">\n"
           "      </compartment>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\">\n"
           "      </species>\n"
           "      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\">\n"
           "      </species>\n"
           "      <species id=\"S3\" name=\"S3\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\">\n"
           "      </species>\n"
           "    </listOfSpecies>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"R1\" name=\"R1\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> compartment </ci>\n"
           "              <ci> k1 </ci>\n"
           "              <ci> S1 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "          <listOfLocalParameters>\n"
           "            <localParameter id=\"k1\" name=\"k1\" value=\"0.1\"/>\n"
           "          </listOfLocalParameters>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction  id=\"R2\" name=\"R2\" reversible=\"false\" fast=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> compartment </ci>\n"
           "              <ci> k1 </ci>\n"
           "              <ci> S2 </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "          <listOfLocalParameters>\n"
           "            <localParameter id=\"k1\" name=\"k1\" value=\"0.1\"/>\n"
           "          </listOfLocalParameters>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string ModelWithLocalParameters::modelName() {
    return "ModelWithLocalParameters";
}


std::string BimolecularEnd::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
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
}

std::unordered_map<std::string, double> BimolecularEnd::steadyState() {
    return std::unordered_map<std::string, double>{
            {"S1", 13.3940},
            {"S2", 8.07426},
            {"S3", 2.72327}
    };
}

std::unordered_map<std::string, rr::Setting> BimolecularEnd::steadyStateSettings() {
    return std::unordered_map<std::string, rr::Setting>{};
}

std::unordered_map<std::string, double> BimolecularEnd::steadyStateFluxes() {
    return std::unordered_map<std::string, double>{
            {"J0", 0.717175},
            {"J1", 1.03483},
            {"J2", 0.717175},
            {"J3", 0.399521},
            {"J4", 0.317655}
    };
}

std::unordered_map<std::string, rr::Setting> BimolecularEnd::jacobianSettings() {
    return std::unordered_map<std::string, rr::Setting>{
            {"steady_state", true},
    };
}

rr::Matrix<double> BimolecularEnd::fullJacobianConc() {
    // copasi generated steady state full jacobian
    rr::Matrix<double> mat({
                                   {-7.730519332005849, 37.545322537461026, 12.768450085549887},
                                   {7.609999999602059,  -38.03670015738035, -12.478993111182907},
                                   {7.683688395905918,  -37.099777221335,   -12.952662332648718},
                           });
    mat.setRowNames({"S1", "S3", "S2"});
    mat.setColNames({"S1", "S3", "S2"});
    return mat;
}

/**
 * Amt is same as conc because volume of single compartment == 1
 */
rr::Matrix<double> BimolecularEnd::fullJacobianAmt() {
    return fullJacobianConc();
}

std::string BimolecularEnd::modelName() {
    return "BimolecularEnd";
}

rr::Matrix<double> BimolecularEnd::linkMatrix() {
    rr::Matrix<double> mat({
                                   {1, 0, 0},
                                   {0, 1, 0},
                                   {0, 0, 1},
                           });
    mat.setColNames({"S1", "S3", "S2"});
    return mat;
}

rr::Matrix<double> BimolecularEnd::NrMatrix() {
    return rr::Matrix<double>({
                                      {1, -1, 0,  0,  1},
                                      {0, 0,  1,  -1, -1},
                                      {0, 1,  -1, 0,  -1},
                              });
}

rr::Matrix<double> BimolecularEnd::KMatrix() {
    return rr::Matrix<double>({
                                      {1,    0},
                                      {0,    1},
                                      {-0.5, 0.5},
                                      {0.5,  0.5},
                                      {0.5,  0.5},
                              });
}

rr::Matrix<double> BimolecularEnd::reducedStoicMatrix() {
    return rr::Matrix<double>({
                                      {1, -1, 0,  0,  1},
                                      {0, 0,  1,  -1, -1},
                                      {0, 1,  -1, 0,  -1},
                              });
}

rr::Matrix<double> BimolecularEnd::fullStoicMatrix() {
    return rr::Matrix<double>({
                                      {1, -1, 0,  0,  1},
                                      {0, 0,  1,  -1, -1},
                                      {0, 1,  -1, 0,  -1},
                              });
}

rr::Matrix<double> BimolecularEnd::extendedStoicMatrix() {
    return rr::Matrix<double>({
                                      {1,  -1, 0,  0,  1},
                                      {0,  1,  -1, 0,  -1},
                                      {0,  0,  1,  -1, -1},
                                      {-1, 0,  0,  0,  0},
                                      {0,  0,  0,  1,  0},
                              });
}

rr::Matrix<double> BimolecularEnd::L0Matrix() {
    return rr::Matrix<double>();
}

rr::Matrix<double> BimolecularEnd::conservationMatrix() {
    return rr::Matrix<double>();
}

rr::Matrix<double> BimolecularEnd::unscaledConcentrationControlCoefficientMatrix() {
    return rr::Matrix<double>({
                                      {14.673,  -6.20668,  -2.24064, -6.22567,  -0.0189895},
                                      {4.90579, -0.794829, -3.27825, -0.832715, -0.037886},
                                      {1.32614, -0.981005, 0.653527, -0.998665, -0.0176601},
                              });
}

rr::Matrix<double> BimolecularEnd::scaledConcentrationControlCoefficientMatrix() {
    return rr::Matrix<double>({
                                      {0.785658, -0.479532, -0.119974, -0.185701,  -0.000450359},
                                      {0.435744, -0.101869, -0.291182, -0.0412034, -0.0014905},
                                      {0.349241, -0.372777, 0.172107,  -0.14651,   -0.00205995},
                              });
}

rr::Matrix<double> BimolecularEnd::unscaledFluxControlCoefficientMatrix() {
    return rr::Matrix<double>({
                                      {0.31285,   0.290665,   0.104931,  0.291554,  0.000889297},
                                      {0.56492,   0.626291,   0.179909,  -0.371121, 0.002588},
                                      {0.31285,   0.290665,   0.104931,  0.291554,  0.000889297},
                                      {0.0607802, -0.0449617, 0.0299526, 0.954229,  -0.000809402},
                                      {0.25207,   0.335626,   0.0749784, -0.662675, 0.0016987},

                              });
}

rr::Matrix<double> BimolecularEnd::scaledFluxControlCoefficientMatrix() {
    return rr::Matrix<double>({
                                      {0.31285,  0.419407,  0.104931,  0.162418, 0.000393892},
                                      {0.391511, 0.626291,  0.124684,  -0.14328, 0.000794419},
                                      {0.31285,  0.419407,  0.104931,  0.162418, 0.000393892},
                                      {0.109106, -0.116459, 0.0537676, 0.954229, -0.000643546},
                                      {0.569104, 1.09338,   0.16928,   -0.83346, 0.0016987},
                              });
}

rr::Matrix<double> BimolecularEnd::unscaledElasticityMatrix() {
    return rr::Matrix<double>({
                                      {-0.0468309, 0,         0},
                                      {0.0736884,  0,         -0.105245},
                                      {0,          -0.445545, 0.184212},
                                      {0,          0.045832,  0},
                                      {-7.61,      37.5453,   12.6632},
                              });
}

rr::Matrix<double> BimolecularEnd::scaledElasticityMatrix() {
    return rr::Matrix<double>({
                                      {-0.874617, 0,        0},
                                      {0.953764,  0,        -0.821173},
                                      {0,         -1.69183, 2.07394},
                                      {0,         0.312409, 0},
                                      {-320.878,  321.878,  321.878},
                              });
}

std::unordered_map<std::string, rr::Setting> BimolecularEnd::mcaSettings() {
    return std::unordered_map<std::string, rr::Setting>();
}

std::string BatchImmigrationDeath03::str() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n"
           "  <model id=\"BatchImmigrationDeath03\" name=\"Batch Immigration-Death (004), variant 03\" substanceUnits=\"item\" timeUnits=\"second\" volumeUnits=\"litre\">\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"Cell\" spatialDimensions=\"3\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"X\" compartment=\"Cell\" initialAmount=\"0\" hasOnlySubstanceUnits=\"true\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"Alpha\" value=\"1\" constant=\"true\"/>\n"
           "      <parameter id=\"Mu\" value=\"4\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"Immigration\" reversible=\"false\">\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"X\" stoichiometry=\"100\" constant=\"false\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <ci> Alpha </ci>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "      <reaction id=\"Death\" reversible=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"X\" stoichiometry=\"1\" constant=\"false\"/>\n"
           "        </listOfReactants>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply>\n"
           "              <times/>\n"
           "              <ci> Mu </ci>\n"
           "              <ci> X </ci>\n"
           "            </apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

std::string BatchImmigrationDeath03::modelName(){
    return "BatchImmigrationDeath03";
}


std::vector<std::string> getAvailableTestModels() {
    return std::vector<std::string>(
            {
                    "SimpleFlux",
                    "SimpleFluxManuallyReduced",
                    "OpenLinearFlux",
                    "Model269",
                    "Model28",
                    "CeilInRateLaw",
                    "FactorialInRateLaw",
                    "Venkatraman2010",
                    "Brown2004",
                    "LayoutOnly",
                    "ModelWithLocalParameters",
                    "BimolecularEnd",
                    "BatchImmigrationDeath03"
            });
}


TestModel *TestModelFactory(const std::string &modelName) {
    if (modelName == "SimpleFlux") {
        return new SimpleFlux();
    } else if (modelName == "Model269") {
        return new Model269();
    } else if (modelName == "Model28") {
        return new Model28();
    } else if (modelName == "CeilInRateLaw") {
        return new CeilInRateLaw();
    } else if (modelName == "FactorialInRateLaw") {
        return new FactorialInRateLaw();
    } else if (modelName == "Venkatraman2010") {
        return new Venkatraman2010();
    } else if (modelName == "OpenLinearFlux") {
        return new OpenLinearFlux();
    } else if (modelName == "SimpleFluxManuallyReduced") {
        return new SimpleFluxManuallyReduced();
    } else if (modelName == "Brown2004") {
        return new Brown2004();
    } else if (modelName == "LayoutOnly") {
        return new LayoutOnly();
    } else if (modelName == "ModelWithLocalParameters") {
        return new ModelWithLocalParameters();
    } else if (modelName == "BimolecularEnd") {
        return new BimolecularEnd();
    } else if (modelName == "BatchImmigrationDeath03") {
        return new BatchImmigrationDeath03();
    } else {
        std::ostringstream err;
        err << "TestModelFactory::TestModelFactory(): no model called \"" << modelName << "\" found. ";
        err << "Available test models include: ";
        for (const auto &name: getAvailableTestModels()) {
            err << "\"" << name << "\", ";
        }
        throw std::runtime_error(err.str());
    }
}


namespace privateSwigTests_ {
    // this section exists only to test the swig bindings
    // and make sure the typemaps are doing what they are supposed
    // to be. Users should completely ignore this

    DoublePair *_testDoublePair(double first, double second) {
        DoublePair *pair = new DoublePair(first, second);
        return pair;
    }

    std::unordered_map<double, double> *_testDoubleMap(double first, double second) {
        std::unordered_map<double, double> *map = new std::unordered_map<double, double>{
                {first, second}
        };
        return map;
    }

    std::unordered_map<std::string, rr::Setting> *_testVariantMap() {
        std::unordered_map<std::string, rr::Setting> *map = new std::unordered_map<std::string, rr::Setting>{
                {"mapsy", rr::Setting(5)}
        };
        return map;
    }

    rr::Setting *_testVariant() {
        rr::Setting *x = new rr::Setting(5.4);
        return x;
    }

    StringDoublePairMap _testResultMap() {
        return StringDoublePairMap{
                {"First", DoublePair(0.5, 1.6)},
        };
    }

    std::vector<std::complex<double>> _testStdComplexZeroImagPart() {
        return std::vector<std::complex<double>>({{2, 0}});
    }

    std::vector<std::complex<double>> _testStdComplexNonZeroImagPart() {
        return std::vector<std::complex<double>>({{3, 4}});
    }

    std::vector<double> _testDoubleVectorTo1DNumpyArray() {
        return std::vector<double>({0.1, 0.2, 0.3});
    }

    rr::Matrix3D<double, double> _testMatrix3D_3x2x3() {
        return rr::Matrix3D<double, double>(
                {0, 3.5, 7.9},
                {
                        {
                                {0.1, 0.2, 0.3},
                                {0.4, 0.5, 0.6},
                        },
                        {
                                {0.7, 0.8, 0.9},
                                {1.0, 1.1, 1.2},
                        },
                        {
                                {1.3, 1.4, 1.5},
                                {1.6, 1.7, 1.8},
                        },
                }
        );
    }

    rr::Matrix3D<double, double> _testMatrix3D_2x3x4() {
        return rr::Matrix3D<double, double>(
                {0, 4.23},
                {
                        {
                                {0,  1,  2,  3},
                                {4,  5,  6,  7},
                                {8,  9,  10, 11},
                        },
                        {
                                {12, 13, 14, 15},
                                {16, 17, 18, 19},
                                {20, 21, 22, 23},
                        },
                }
        );
    }

    rr::Matrix3D<double, double> _testMatrix3D_4x3x2() {
        return rr::Matrix3D<double, double>(
                {0, 6, 12, 18},
                {
                        {
                                {0,  1},
                                {2,  3},
                                {4,  5},
                        },
                        {
                                {6,  7},
                                {8,  9},
                                {10, 11},
                        },
                        {
                                {12, 13},
                                {14, 15},
                                {16, 17},
                        },
                        {
                                {18, 19},
                                {20, 21},
                                {22, 23},
                        }
                }
        );
    }

    rr::Matrix3D<double, double> _testMatrix3D_3x4x2() {
        return rr::Matrix3D<double, double>(
                {0, 6, 12},
                {
                        {
                                {0,  1},
                                {2,  3},
                                {4,  5},
                                {6,  7},
                        },
                        {
                                {8,  9},
                                {10, 11},
                                {12, 13},
                                {14, 15},
                        },
                        {
                                {16, 17},
                                {18, 19},
                                {20, 21},
                                {22, 23},
                        },
                }
        );
    }

    ls::Matrix<double> _testLsMatrixWithLabels() {
        ls::Matrix<double> lsMatrix({
                                            {1.1, 2.2}
                                    });
        lsMatrix.setColNames({"C0"});
        lsMatrix.setRowNames({"R0", "R1"});
        return lsMatrix;
    };

    rr::Matrix<double> _testRRMatrixWithLabels() {
        rr::Matrix<double> rrMatrix({
                                            {1.1, 2.2}
                                    });
        rrMatrix.setColNames({"C0"});
        rrMatrix.setRowNames({"R0", "R1"});
        return rrMatrix;
    };

    std::string _testPythonStringToCxxRoundTrip(std::string s){
        return s;
    }

    std::vector<std::string> _testAddElementToStringVec(std::vector<std::string> stringVec, std::string newElement){
        stringVec.push_back(newElement);
        return stringVec;
    }

     std::vector<std::string>& _testAddElementToStringVecAsReference( std::vector< std::string>& stringVec, std::string newElement){
        stringVec.push_back(newElement);
        return stringVec;
    }


}


#include "rr/rrRoadRunner.h"
#include "rr/Integrator.h"
#include "rr/Setting.h"
#include <iostream>
#include <string>

using namespace rr;

// S1 -> S2 with a first order rate: a model with a sparsity pattern of the Jacobian.
static std::string getSBMLString() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n"
           "  <model id=\"klu\">\n"
           "    <listOfCompartments>\n"
           "      <compartment id=\"c\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
           "    </listOfCompartments>\n"
           "    <listOfSpecies>\n"
           "      <species id=\"S1\" compartment=\"c\" initialAmount=\"1\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "      <species id=\"S2\" compartment=\"c\" initialAmount=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
           "    </listOfSpecies>\n"
           "    <listOfParameters>\n"
           "      <parameter id=\"k1\" value=\"1\" constant=\"true\"/>\n"
           "    </listOfParameters>\n"
           "    <listOfReactions>\n"
           "      <reaction id=\"r1\" reversible=\"false\">\n"
           "        <listOfReactants>\n"
           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfReactants>\n"
           "        <listOfProducts>\n"
           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
           "        </listOfProducts>\n"
           "        <kineticLaw>\n"
           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
           "            <apply><times/><ci> k1 </ci><ci> S1 </ci></apply>\n"
           "          </math>\n"
           "        </kineticLaw>\n"
           "      </reaction>\n"
           "    </listOfReactions>\n"
           "  </model>\n"
           "</sbml>";
}

// Simulates with the sparse difference quotient Jacobian and the sparse linear solver KLU, and
// prints whether CVODE used KLU.
int main() {
    RoadRunner roadRunner(getSBMLString());
    Integrator *integrator = roadRunner.getIntegrator();
    integrator->setValue("jacobian", Setting(std::string("sparse_difference_quotient")));
    integrator->setValue("linear_solver", Setting(std::string("klu")));
    roadRunner.simulate(0, 1, 11);
    const bool klu = integrator->getStatistics()["sparse_linear_solver"].getAs<bool>();
    std::cout << "KLU in use: " << (klu ? "yes" : "no") << std::endl;
    return klu ? 0 : 1;
}

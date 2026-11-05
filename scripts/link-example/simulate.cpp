/**
 * Minimal roadrunner simulation, used to show how to link a C++ program
 * against a roadrunner built by scripts/build_roadrunner.sh.
 *
 * The model is a one parameter exponential decay, dx/dt = -k * x with
 * x(0) = 1 and k = 1, so the result can be checked against exp(-t) and the
 * program doubles as a smoke test of the install tree. It is passed as a
 * string, so the example needs no data files.
 *
 * The headers live in <install>/include/rr, and the include directory that
 * the imported cmake target hands over is <install>/include, hence "rr/...".
 */
#include "rr/rrRoadRunner.h"
#include "rr/rrException.h"
#include "rr/rrVersionInfo.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

    const char *const DECAY_MODEL = R"SBML(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay">
    <listOfParameters>
      <parameter id="x" value="1" constant="false"/>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <times/>
            <apply><minus/><ci> k </ci></apply>
            <ci> x </ci>
          </apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>
)SBML";

}  // namespace

int main() {
    try {
        rr::RoadRunner runner;
        runner.load(DECAY_MODEL);
        runner.setSelections(std::vector<std::string>{"time", "x"});

        // start = 0, duration = 1, 11 output points
        const ls::DoubleMatrix *result = runner.simulate(0.0, 1.0, 11);

        for (const std::string &name : result->getColNames()) {
            std::cout << std::setw(12) << name;
        }
        std::cout << '\n' << std::fixed << std::setprecision(6);
        for (unsigned int row = 0; row < result->numRows(); ++row) {
            for (unsigned int col = 0; col < result->numCols(); ++col) {
                std::cout << std::setw(12) << (*result)(row, col);
            }
            std::cout << '\n';
        }

        const double simulated = runner.getValue("x");
        const double expected = std::exp(-1.0);
        std::cout << "\nx(1) = " << simulated << ", exp(-1) = " << expected << '\n';
        if (std::abs(simulated - expected) > 1e-5) {
            std::cerr << "the result is off by more than 1e-5\n";
            return 1;
        }
        std::cout << "roadrunner " << rr::getVersionStr() << " works\n";
        return 0;
    } catch (const rr::Exception &error) {
        std::cerr << "roadrunner failed: " << error.getMessage() << '\n';
        return 1;
    }
}

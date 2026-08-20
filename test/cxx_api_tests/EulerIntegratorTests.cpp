#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "RoadRunnerTest.h"
#include "EulerIntegrator.h"

using namespace rr;

// One species, no events. Small enough that the Euler integrator's
// internal buffers end up sized for exactly one state variable and zero
// events.
static const std::string OneSpeciesNoEventSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="one_species">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="S0" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
 <listOfReactions>
  <reaction id="R0" reversible="false">
   <listOfReactants><speciesReference species="S0"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S0</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

// Five species and one event. Reloading into this model after the one
// above should force the Euler integrator to rebuild its state/rate
// buffers (now sized for 5 states) and its event-status vector (now sized
// for 1 event), and drop its pointer to the model RoadRunner::load() just
// replaced.
static const std::string FiveSpeciesWithEventSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="five_species">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="A" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"/>
  <species id="B" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"/>
  <species id="C" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"/>
  <species id="D" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"/>
  <species id="E" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
 <listOfReactions>
  <reaction id="R0" reversible="false">
   <listOfReactants><speciesReference species="A"/></listOfReactants>
   <listOfProducts><speciesReference species="B"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
 <listOfEvents>
  <event id="ev0" useValuesFromTriggerTime="true">
   <trigger>
    <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><lt/><ci>A</ci><cn>0.9</cn></apply></math>
   </trigger>
   <listOfEventAssignments>
    <eventAssignment variable="E">
     <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>99</cn></math>
    </eventAssignment>
   </listOfEventAssignments>
  </event>
 </listOfEvents>
</model></sbml>)";

class EulerIntegratorTests : public RoadRunnerTest {
public:
    EulerIntegratorTests() = default;
};

/**
 * Regression test for EulerIntegrator inheriting Integrator's empty
 * syncWithModel(). RoadRunner::load() deletes the old ExecutableModel and
 * calls syncWithModel() on every already-constructed integrator so they can
 * pick up the new one and resize their internal state. Euler didn't
 * override it, so after a reload it kept a dangling pointer to the freed
 * model plus rate/state buffers and an event-status vector sized for the
 * model that no longer exists.
 *
 * Under a plain (non-sanitized) build this may silently produce wrong
 * results rather than crash outright -- the reliable signal is a
 * heap-use-after-free / heap-buffer-overflow under ASan or valgrind. The
 * functional assertion below (the event never fires) is a bug that doesn't
 * depend on a sanitizer to observe, but running this test under ASan is
 * the more decisive check for the underlying memory-safety fix.
 */
TEST_F(EulerIntegratorTests, ReloadResizesBuffersAndEventsWithoutStaleModel) {
    RoadRunner rr(OneSpeciesNoEventSBML);
    rr.setIntegrator("euler");

    // Run once so Euler actually allocates its buffers/event vector
    // against the 1-species/0-event model.
    rr.simulate(0, 1, 10);

    rr.load(FiveSpeciesWithEventSBML);
    ASSERT_EQ("euler", rr.getIntegrator()->getName())
        << "reload should not have reset the selected integrator";

    ASSERT_NO_THROW({
        rr.simulate(0, 20, 200);
    });

    // A decays below 0.9 well within this window, so the event should have
    // fired and set E = 99. If Euler's event-status vector never grew past
    // its stale size of zero (carried over from the event-free model), the
    // trigger can never be observed and E stays at its initial value of 1.
    double finalE = rr.getValue("E");
    EXPECT_NEAR(99.0, finalE, 1e-6)
        << "event did not fire after reload; Euler's event bookkeeping "
           "likely still reflects the pre-reload model.";
}

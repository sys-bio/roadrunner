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
#include "llvm/LLVMException.h"
#include "llvm/LLVMExecutableModel.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
  path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
  SBMLFeatures() = default;
};

TEST_F(SBMLFeatures, named_stoich_list) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  vector<string> stoichs = rr.getStoichiometryIds();
  EXPECT_STREQ(stoichs[0].c_str(), "stoich(A, J0)");
  EXPECT_STREQ(stoichs[1].c_str(), "n");
  EXPECT_STREQ(stoichs[2].c_str(), "m");
  EXPECT_STREQ(stoichs[3].c_str(), "q");
}


TEST_F(SBMLFeatures, issue1306_named_stoich_value) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  EXPECT_EQ(rr.getValue("q"), 7.0);
  // stoich(species, reaction) reads the raw stoichiometry-matrix cell:
  // negative for reactants (B, via "n"), positive for products (C via "m", D via "q").
  EXPECT_EQ(rr.getValue("stoich(B, J0)"), -3.0);
  EXPECT_EQ(rr.getValue("stoich(C, J0)"), 5.0);
  EXPECT_EQ(rr.getValue("stoich(D, J0)"), 7.0);
  EXPECT_EQ(rr.getValue("J0"), 15.0);
  rr.oneStep(0, 0.01);
  EXPECT_NEAR(rr.getValue("A"), 0.8607083324139845, 0.00001);
  EXPECT_NEAR(rr.getValue("B"), 6.582124997241953, 0.00001);
  EXPECT_NEAR(rr.getValue("C"), 0.6964583379300783, 0.00001);
  EXPECT_NEAR(rr.getValue("D"), 0.9750416731021092, 0.00001);
  string sbml = rr.getCurrentSBML();

}


TEST_F(SBMLFeatures, issue1306_named_stoich_steadyState) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.steadyState();
  EXPECT_NEAR(rr.getValue("B"), 6.0, 0.00001);
  rr.reset();
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  rr.setConservedMoietyAnalysis(true);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  //EXPECT_EQ(rr.getValue("q"), 7.0);
  //EXPECT_EQ(rr.getValue("J0"), 0.0);
  rr.steadyState();
  EXPECT_NEAR(rr.getValue("B"), 4.0, 0.00001);
}


TEST_F(SBMLFeatures, named_stoich_init_value) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("init(n)", 3);
  rr.setValue("init(m)", 5);
  rr.setValue("init(q)", 7);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  EXPECT_EQ(rr.getValue("q"), 7.0);
  EXPECT_EQ(rr.getValue("init(n)"), 3.0);
  EXPECT_EQ(rr.getValue("init(m)"), 5.0);
  EXPECT_EQ(rr.getValue("init(q)"), 7.0);
}


TEST_F(SBMLFeatures, named_stoich_values) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  EXPECT_EQ(rr.getModel()->getValue("n"), 3.0);
  EXPECT_EQ(rr.getModel()->getValue("m"), 5.0);
  EXPECT_EQ(rr.getModel()->getValue("q"), 7.0);
  EXPECT_EQ(rr.getModel()->getValue("init(n)"), 1.0);
  EXPECT_EQ(rr.getModel()->getValue("init(m)"), 2.0);
  EXPECT_EQ(rr.getModel()->getValue("init(q)"), 3.0);
}


TEST_F(SBMLFeatures, add_rule_to_named_stoich) {
  // "n" is the reactant B's stoichiometry (declared 1); "m" (C, declared 2)
  // and "q" (D, declared 3) are unaffected. Kinetic law rate = (n+m+q)*A.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("n", "5", true));
  EXPECT_EQ(rr.getValue("n"), 5.0);
  // rate = (5 + 2 + 3) * A(1) = 10
  EXPECT_NEAR(rr.getValue("J0"), 10.0, 1e-9);

  // "q" is the product D's stoichiometry (declared 3).
  rr::RoadRunner rr2((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr2.addRateRule("q", "5", true));
  // rate rules haven't integrated yet at t=0; q should still be its declared value.
  EXPECT_NEAR(rr2.getValue("q"), 3.0, 1e-9);
  rr2.oneStep(0, 1.0);
  // dq/dt = 5, so after 1 time unit q should have grown by ~5.
  EXPECT_NEAR(rr2.getValue("q"), 8.0, 1e-6);
}


TEST_F(SBMLFeatures, add_rate_rule_to_reactant_stoich) {
  // "n" is reactant B's stoichiometry (declared 1). Mirrors the "q"
  // (product) rate-rule case in add_rule_to_named_stoich, but exercises the
  // reactant sign-handling path: createStoichiometryNode negates reactant
  // values for the CSR cell, and that same negated value must not leak into
  // the rate-rule integration slot.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("n", "5", true));
  // rate rules haven't integrated yet at t=0; n should still be its declared value.
  EXPECT_NEAR(rr.getValue("n"), 1.0, 1e-9);
  rr.oneStep(0, 1.0);
  // dn/dt = 5, so after 1 time unit n should have grown by ~5, mirroring q's 3->8.
  EXPECT_NEAR(rr.getValue("n"), 6.0, 1e-6);
}


TEST_F(SBMLFeatures, add_assignment_rule_to_reactant_stoich_tracks_time) {
  // "n" is reactant B's stoichiometry. An assignment rule that depends on
  // time must be resynced into the stoichiometry matrix as the simulation
  // progresses, not just read once at t=0.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("n", "1 + time", true));
  EXPECT_NEAR(rr.getValue("n"), 1.0, 1e-9);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("n"), 3.0, 1e-9);
}


TEST_F(SBMLFeatures, set_value_on_multi_reactant_stoich_before_simulate) {
  // A appears twice as a reactant in J0, via two independently named,
  // rule-free species references (r1=2, r2=3). Setting one before
  // simulating must not disturb the other, and the combined (net) cell used
  // in the reaction must reflect the sum of both.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_EQ(r1, 2.0);
  EXPECT_EQ(rr.getValue("r2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("r1", 5));
  EXPECT_EQ(rr.getValue("r1"), 5.0);
  EXPECT_EQ(rr.getValue("r2"), 3.0);

  // stoich(A, J0) reads the raw matrix cell: A is consumed here, so the
  // combined cell is negative: -(r1 + r2) = -8.
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -8.0, 1e-9);

  // dA/dt = -(r1+r2)*k*A < 0, A should be decreasing
  rr.oneStep(0, 0.01);
  EXPECT_LT(rr.getValue("A"), 10.0);
}


TEST_F(SBMLFeatures, multi_product_stoich_collision_set_one_of_two) {
  // B is produced twice in J0, via two independently named, rule-free
  // species references (p1=2, p2=3). Setting one before simulating must
  // not disturb the other, and the combined (net) cell used in the
  // reaction must reflect the sum of both.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_product.xml").string());
  double p1;
  ASSERT_NO_THROW(p1 = rr.getValue("p1"));
  EXPECT_EQ(p1, 2.0);
  EXPECT_EQ(rr.getValue("p2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("p1", 5));
  EXPECT_EQ(rr.getValue("p1"), 5.0);
  EXPECT_EQ(rr.getValue("p2"), 3.0);

  // combined cell should be p1 + p2 = 8 (products are positive, no sign flip)
  EXPECT_NEAR(rr.getValue("stoich(B, J0)"), 8.0, 1e-9);

  rr.oneStep(0, 0.01);
  EXPECT_GT(rr.getValue("B"), 0.0);
}


TEST_F(SBMLFeatures, multi_reactant_product_cross_collision_set_one_of_two) {
  // A appears once as a reactant (x1) and once as a product (x2) in the
  // SAME reaction -- no literal duplicate within either list, but both
  // occurrences still collide on the same (species, reaction) CSR cell
  // (LLVMModelDataSymbols shares one speciesMap across the reactant and
  // product loops). Setting one must not disturb the other. stoich(A, J0)
  // reads the raw matrix cell (-x1 + x2), unaffected by the ambiguity of
  // "reactant or product" that a per-reference read would have.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_mixed.xml").string());
  double x1;
  ASSERT_NO_THROW(x1 = rr.getValue("x1"));
  EXPECT_EQ(x1, 2.0);
  EXPECT_EQ(rr.getValue("x2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("x1", 5));
  EXPECT_EQ(rr.getValue("x1"), 5.0);
  EXPECT_EQ(rr.getValue("x2"), 3.0);

  // net effect: -x1 + x2 = -5 + 3 = -2
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -2.0, 1e-9);

  rr.oneStep(0, 0.01);
  EXPECT_LT(rr.getValue("A"), 10.0);
}


TEST_F(SBMLFeatures, multi_reactant_stoich_set_via_selector_throws) {
  // Unlike getValue(stoich(x,y)), which reads the role-agnostic matrix
  // cell, setValue(stoich(x,y), v) means "set the underlying
  // speciesReference" -- which is ambiguous when more than one reference
  // shares the cell, so it must throw rather than silently pick one.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_THROW(rr.setValue("stoich(A, J0)", 5), rrllvm::LLVMException);
}


TEST_F(SBMLFeatures, named_stoich_selectors) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  SelectionRecord record = rr.createSelection("n");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  EXPECT_STREQ(record.to_string().c_str(), "n");
  EXPECT_EQ(record.index, 1);
  EXPECT_EQ(record.p1, "n");
  EXPECT_EQ(record.p2, "");

  record = rr.createSelection("stoich(A, J0)");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  EXPECT_STREQ(record.to_string().c_str(), "stoich(A, J0)");
  EXPECT_EQ(record.index, 0);
  EXPECT_EQ(record.p1, "A");
  EXPECT_EQ(record.p2, "J0");
}


TEST_F(SBMLFeatures, named_stoich_set_and_reset) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic.xml").string());
  rr.getSimulateOptions().setDuration(10);

  const ls::DoubleMatrix run1 = *rr.simulate();

  rr.reset(SelectionRecord::ALL);
  rr.setValue("N", 2);
  const ls::DoubleMatrix run2 = *rr.simulate();

  rr.reset(SelectionRecord::ALL);
  const ls::DoubleMatrix run3 = *rr.simulate();

  ASSERT_EQ(run1.numRows(), run2.numRows());
  ASSERT_EQ(run1.numRows(), run3.numRows());
  for (int i = 0; i < run1.numRows(); i++) {
    EXPECT_EQ(run1(i, 1), run3(i, 1));
    EXPECT_NEAR(run2(i, 1), 2 * run1(i, 1), 1e-12);
  }
}


TEST_F(SBMLFeatures, get_named_stoich_value_from_model) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  ExecutableModel* em = rr.getModel();
  rrllvm::LLVMExecutableModel* llem = static_cast<rrllvm::LLVMExecutableModel*>(em);
  
  EXPECT_EQ(llem->getValue("n"), 1);
  llem->setValue("n", 3);
  EXPECT_EQ(llem->getValue("n"), 3);

  // getValue(stoich(x,y)) reads the raw (role-agnostic) matrix cell, so A
  // being a reactant reads as -1. setValue(stoich(x,y), v), by contrast,
  // means "set the underlying speciesReference" -- it takes v as a
  // positive magnitude and sign-flips internally for a reactant, same as
  // the named-id form. Setting 5 here therefore stores -5.
  EXPECT_EQ(llem->getValue("stoich(A, J0)"), -1);
  llem->setValue("stoich(A, J0)", 5);
  EXPECT_EQ(llem->getValue("stoich(A, J0)"), 5);
}


// init(x) and x currently share the same storage for stoichiometries, which
// is a bug: setting one should not affect the other. These three tests pin
// down the expected, independent behavior for the three cases that need it:
// a plain named stoichiometry, a named stoichiometry whose species is a
// MultiSpeciesReference, and an unnamed stoichiometry accessed via
// stoich(species, reaction).

TEST_F(SBMLFeatures, named_stoich_init_and_current_are_independent) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());

  rr.setValue("init(n)", 3);
  rr.setValue("n", 5);

  EXPECT_EQ(rr.getValue("init(n)"), 3);
  EXPECT_EQ(rr.getValue("n"), 5);
}

TEST_F(SBMLFeatures, named_stoich_init_and_current_are_independent_multi_reactant) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());

  ASSERT_NO_THROW(rr.setValue("init(r1)", 3));
  ASSERT_NO_THROW(rr.setValue("r1", 5));

  EXPECT_EQ(rr.getValue("init(r1)"), 3);
  EXPECT_EQ(rr.getValue("r1"), 5);
}

TEST_F(SBMLFeatures, unnamed_stoich_init_and_current_are_independent) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());

  // A is an unnamed reactant in J0 (stoichiometry 1). setValue(stoich(x,y), v)
  // takes v as a positive magnitude and sign-flips internally for a reactant,
  // so 3/5 here become -3/-5.
  ASSERT_NO_THROW(rr.setValue("init(stoich(A, J0))", 3));
  ASSERT_NO_THROW(rr.setValue("stoich(A, J0)", 5));

  EXPECT_EQ(rr.getValue("init(stoich(A, J0))"), 3);
  EXPECT_EQ(rr.getValue("stoich(A, J0)"), 5);

  //For B, its stoichiometry is named 'n', and it is a reactant
  // When setting this using the 'stoich(B, J0)' form, internally, this
  // sets the literal stoichiometry of the reactant to a negative value.

  ASSERT_NO_THROW(rr.setValue("init(stoich(B, J0))", 3));
  ASSERT_NO_THROW(rr.setValue("stoich(B, J0)", 5));

  EXPECT_EQ(rr.getValue("init(stoich(B, J0))"), 3);
  EXPECT_EQ(rr.getValue("stoich(B, J0)"), 5);

  EXPECT_EQ(rr.getValue("init(n)"), -3);
  EXPECT_EQ(rr.getValue("n"), -5);

  //Conversely, when setting this using the 'n' form, internally, this
  // sets the literal stoichiometry of the reactant to a positive value,
  // meaning that the 'stoich(B, J0)' form will be negative (like the 
  // stoichiometry matrix)

  ASSERT_NO_THROW(rr.setValue("init(n)", 4));
  ASSERT_NO_THROW(rr.setValue("n", 6));

  EXPECT_EQ(rr.getValue("init(stoich(B, J0))"), -4);
  EXPECT_EQ(rr.getValue("stoich(B, J0)"), -6);

  EXPECT_EQ(rr.getValue("init(n)"), 4);
  EXPECT_EQ(rr.getValue("n"), 6);

  // For C, its stoichiometry is named 'm', and it is a product.
  // This means the regardless of form, there are never any sign flips

  ASSERT_NO_THROW(rr.setValue("init(stoich(C, J0))", 3));
  ASSERT_NO_THROW(rr.setValue("stoich(C, J0)", 5));

  EXPECT_EQ(rr.getValue("init(stoich(C, J0))"), 3);
  EXPECT_EQ(rr.getValue("stoich(C, J0)"), 5);

  EXPECT_EQ(rr.getValue("init(m)"), 3);
  EXPECT_EQ(rr.getValue("m"), 5);

  ASSERT_NO_THROW(rr.setValue("init(m)", 4));
  ASSERT_NO_THROW(rr.setValue("m", 6));

  EXPECT_EQ(rr.getValue("init(stoich(C, J0))"), 4);
  EXPECT_EQ(rr.getValue("stoich(C, J0)"), 6);

  EXPECT_EQ(rr.getValue("init(m)"), 4);
  EXPECT_EQ(rr.getValue("m"), 6);

}


// Finds the speciesReference for the given species in a reaction, whether
// it's named or not -- used to check stoichiometry values in SBML text
// parsed back via libSBML, rather than by regexing the string directly.
static libsbml::SpeciesReference* findSpeciesReference(libsbml::Reaction* rxn, const string& speciesId) {
  for (unsigned int i = 0; i < rxn->getNumReactants(); i++) {
    libsbml::SpeciesReference* r = rxn->getReactant(i);
    if (r->getSpecies() == speciesId) {
      return r;
    }
  }
  for (unsigned int i = 0; i < rxn->getNumProducts(); i++) {
    libsbml::SpeciesReference* p = rxn->getProduct(i);
    if (p->getSpecies() == speciesId) {
      return p;
    }
  }
  return nullptr;
}


TEST_F(SBMLFeatures, named_stoich_reflected_in_current_sbml) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());

  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);

  string sbml = rr.getCurrentSBML();
  libsbml::SBMLDocument* doc = libsbml::readSBMLFromString(sbml.c_str());
  libsbml::Reaction* rxn = doc->getModel()->getReaction("J0");

  // "n"/"m"/"q" are set and read as the reference's own literal value, with
  // no sign correction for reactant/product, so the raw SBML attribute
  // should match exactly.
  EXPECT_EQ(findSpeciesReference(rxn, "B")->getStoichiometry(), 3.0);
  EXPECT_EQ(findSpeciesReference(rxn, "C")->getStoichiometry(), 5.0);
  EXPECT_EQ(findSpeciesReference(rxn, "D")->getStoichiometry(), 7.0);

  delete doc;

  sbml = rr.getSBML();
  doc = libsbml::readSBMLFromString(sbml.c_str());
  rxn = doc->getModel()->getReaction("J0");

  // Ensure original values didn't change
  EXPECT_EQ(findSpeciesReference(rxn, "B")->getStoichiometry(), 1.0);
  EXPECT_EQ(findSpeciesReference(rxn, "C")->getStoichiometry(), 2.0);
  EXPECT_EQ(findSpeciesReference(rxn, "D")->getStoichiometry(), 3.0);

  delete doc;
}


TEST_F(SBMLFeatures, named_stoich_init_reflected_in_sbml) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());

  rr.setValue("init(n)", 3);
  rr.setValue("init(m)", 5);
  rr.setValue("init(q)", 7);

  // getSBML() serializes the document directly, which setInitValue mutates
  // in place -- so it reflects init values, not whatever getCurrentSBML()
  // would show.
  string sbml = rr.getSBML();
  libsbml::SBMLDocument* doc = libsbml::readSBMLFromString(sbml.c_str());
  libsbml::Reaction* rxn = doc->getModel()->getReaction("J0");

  EXPECT_EQ(findSpeciesReference(rxn, "B")->getStoichiometry(), 3.0);
  EXPECT_EQ(findSpeciesReference(rxn, "C")->getStoichiometry(), 5.0);
  EXPECT_EQ(findSpeciesReference(rxn, "D")->getStoichiometry(), 7.0);

  delete doc;
}


TEST_F(SBMLFeatures, stoich_selector_set_reflected_in_sbml) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());

  // Init values, set via the the species+reaction selector.
  rr.setValue("init(stoich(A, J0))", 12);
  rr.setValue("init(stoich(B, J0))", 13);
  rr.setValue("init(stoich(C, J0))", 14);
  rr.setValue("init(stoich(D, J0))", 15);

  // Current values.
  rr.setValue("stoich(A, J0)", 2);
  rr.setValue("stoich(B, J0)", 3);
  rr.setValue("stoich(C, J0)", 4);
  rr.setValue("stoich(D, J0)", 5);

  // Current stoichiometry is exported via getCurrentSBML(). A and B are
  // reactants, so the species/reaction selector's matrix-convention value is
  // negated relative to the raw SBML attribute; C and D are products, so
  // there's no sign difference.
  string currentSbml = rr.getCurrentSBML();
  libsbml::SBMLDocument* currentDoc = libsbml::readSBMLFromString(currentSbml.c_str());
  libsbml::Reaction* currentRxn = currentDoc->getModel()->getReaction("J0");

  EXPECT_EQ(findSpeciesReference(currentRxn, "A")->getStoichiometry(), -2.0);
  EXPECT_EQ(findSpeciesReference(currentRxn, "B")->getStoichiometry(), -3.0);
  EXPECT_EQ(findSpeciesReference(currentRxn, "C")->getStoichiometry(), 4.0);
  EXPECT_EQ(findSpeciesReference(currentRxn, "D")->getStoichiometry(), 5.0);

  delete currentDoc;

  // Init stoichiometry is exported via getSBML(), and should reflect the
  // separately-set init values above, independent of the current ones.
  string initSbml = rr.getSBML();
  libsbml::SBMLDocument* initDoc = libsbml::readSBMLFromString(initSbml.c_str());
  libsbml::Reaction* initRxn = initDoc->getModel()->getReaction("J0");

  EXPECT_EQ(findSpeciesReference(initRxn, "A")->getStoichiometry(), -12.0);
  EXPECT_EQ(findSpeciesReference(initRxn, "B")->getStoichiometry(), -13.0);
  EXPECT_EQ(findSpeciesReference(initRxn, "C")->getStoichiometry(), 14.0);
  EXPECT_EQ(findSpeciesReference(initRxn, "D")->getStoichiometry(), 15.0);

  delete initDoc;
}


// Boundary species never get a row in the stoichiometry matrix, so a named 
// reference to one has no matrix cell to belong to -- it's purely an 
// independently-stored value, with none of the collision/delta machinery
// MultiSpeciesReference needs.

TEST_F(SBMLFeatures, named_boundary_stoich_value) {
  // J2: S1 (floating reactant, stoich 1, unnamed) -> m=X (boundary product,
  // stoich 2). X is boundary, so "m" has no stoichiometry-matrix cell.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_boundary_species.xml").string());
  EXPECT_EQ(rr.getValue("m"), 2.0);
  rr.setValue("m", 7);
  EXPECT_EQ(rr.getValue("m"), 7.0);
}

TEST_F(SBMLFeatures, named_boundary_stoich_init_value) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_boundary_species.xml").string());
  rr.setValue("init(m)", 9);
  EXPECT_EQ(rr.getValue("m"), 9.0);
  EXPECT_EQ(rr.getValue("init(m)"), 9.0);
}

TEST_F(SBMLFeatures, named_boundary_stoich_init_and_current_are_independent) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_boundary_species.xml").string());
  rr.setValue("init(m)", 3);
  rr.setValue("m", 5);
  EXPECT_EQ(rr.getValue("init(m)"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
}

TEST_F(SBMLFeatures, named_boundary_stoich_no_species_reaction_form) {
  // stoich(species, reaction) addresses a matrix cell, and boundary species
  // don't have one -- X's stoichiometry is only reachable by its own name.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_boundary_species.xml").string());
  EXPECT_THROW(rr.createSelection("stoich(X, J2)"), Exception);
}

TEST_F(SBMLFeatures, named_boundary_stoich_used_in_kinetic_law) {
  // J2: m=X (boundary reactant, stoich 2) -> S1 (floating product, unnamed).
  // kineticLaw = 5 * X^m. Exercises "m" resolving to its own raw value when
  // read as a term inside another formula, not just via getValue("m").
  rr::RoadRunner rr((SBMLFeaturesDir / "named_boundary_species_in_kl.xml").string());
  EXPECT_EQ(rr.getValue("m"), 2.0);
  // rate = 5 * 5^2 = 125
  EXPECT_NEAR(rr.getValue("J2"), 125.0, 1e-9);
  rr.setValue("m", 3);
  // rate = 5 * 5^3 = 625
  EXPECT_NEAR(rr.getValue("J2"), 625.0, 1e-9);
}

TEST_F(SBMLFeatures, named_boundary_stoich_reflected_in_sbml) {
  RoadRunner rr((SBMLFeaturesDir / "named_boundary_species.xml").string());

  rr.setValue("m", 7);
  string sbml = rr.getCurrentSBML();
  libsbml::SBMLDocument* doc = libsbml::readSBMLFromString(sbml.c_str());
  libsbml::Reaction* rxn = doc->getModel()->getReaction("J2");
  EXPECT_EQ(findSpeciesReference(rxn, "X")->getStoichiometry(), 7.0);
  delete doc;

  rr.setValue("init(m)", 9);
  sbml = rr.getSBML();
  doc = libsbml::readSBMLFromString(sbml.c_str());
  rxn = doc->getModel()->getReaction("J2");
  EXPECT_EQ(findSpeciesReference(rxn, "X")->getStoichiometry(), 9.0);
  delete doc;
}



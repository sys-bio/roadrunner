import unittest
import rrplugins


class AutoPluginTests(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @unittest.skip("Doesn't work on non-Windows platforms.")
    def test_running_auto(self):
        model = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Created by libAntimony version v3.1.0 with libSBML version 5.20.5. -->
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="__main" id="__main">
    <listOfCompartments>
      <compartment sboTerm="SBO:0000410" id="default_compartment" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="default_compartment" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="default_compartment" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="_J0" reversible="true">
        <listOfReactants>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S2" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k1 </ci>
              <ci> S1 </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
        auto = rrplugins.Plugin("tel_auto2000")
        auto.setProperty("SBML", model)
        auto.execute()


if __name__ == "__main__":
    unittest.main()

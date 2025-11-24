import unittest
import rrplugins

try:
    from roadrunner.tests import TestModelFactory as tmf
    from roadrunner.tests.RoadRunnerTest import RoadRunnerTest
except:
    import TestModelFactory as tmf
    from RoadRunnerTest import RoadRunnerTest

class AutoPluginTests(RoadRunnerTest):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def loadTestModel(self, modelName: str):
        """Instantiate instance of TestModel called modelName"""
        self.checkValidTestModelName(modelName)
        testModel = tmf.TestModelFactory(modelName)
        self.checkTestModelImplements(testModel, tmf.StructuralProperties)
        return testModel

    def test_running_auto(self):
        testModel = self.loadTestModel("BimolecularEnd")
        auto = rrplugins.Plugin("tel_auto2000")
        auto.setProperty("SBML", testModel.str())
        auto.execute()


if __name__ == "__main__":
    unittest.main()

import roadrunner.*;
// import roadrunner.RoadRunner;
// import org.la4j.Matrices;
import org.la4j.Matrix;

public class roadrunner_java_sim_test1 {
  public static void main(String argv[]) {
    String decayModel = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<!-- Created by XMLPrettyPrinter on 7/11/2014 from JDesigner 2.4.32 -->\r\n<sbml xmlns = \"http://www.sbml.org/sbml/level2\" level = \"2\" version = \"1\" xmlns:jd2 = \"http://www.sys-bio.org/sbml/jd2\">\r\n   <!--                     -->\r\n   <!--  Model Starts Here  -->\r\n   <!--                     -->\r\n   <model id = \"untitled\">\r\n      <listOfCompartments>\r\n         <compartment id = \"compartment\" size = \"1\"/>\r\n      </listOfCompartments>\r\n      <listOfSpecies>\r\n         <species id = \"Node0\" boundaryCondition = \"false\" initialConcentration = \"10\" compartment = \"compartment\"/>\r\n         <species id = \"Node1\" boundaryCondition = \"false\" initialConcentration = \"0\" compartment = \"compartment\"/>\r\n      </listOfSpecies>\r\n      <listOfParameters>\r\n         <parameter id = \"J0_k\" value = \"0.1\"/>\r\n      </listOfParameters>\r\n      <listOfReactions>\r\n         <reaction id = \"J0\" reversible = \"false\">\r\n            <listOfReactants>\r\n               <speciesReference species = \"Node0\" stoichiometry = \"1\"/>\r\n            </listOfReactants>\r\n            <listOfProducts>\r\n               <speciesReference species = \"Node1\" stoichiometry = \"1\"/>\r\n            </listOfProducts>\r\n            <kineticLaw>\r\n               <math xmlns = \"http://www.w3.org/1998/Math/MathML\">\r\n                  <apply>\r\n                     <times/>\r\n                     <ci>\r\n                           J0_k\r\n                     </ci>\r\n                     <ci>\r\n                           Node0\r\n                     </ci>\r\n                  </apply>\r\n               </math>\r\n            </kineticLaw>\r\n         </reaction>\r\n      </listOfReactions>\r\n   </model>\r\n</sbml>\r\n";
    RoadRunner rr = new RoadRunner(decayModel);
    System.out.println(rr.getExtendedVersionInfo());
    System.out.println("- Loaded rr");
    LabeledData x = rr.simulate(0, 10, 10);
    System.out.println("- Simulate called");
    System.out.println(x);
  }
}

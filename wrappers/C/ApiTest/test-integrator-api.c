// == FILEDOC =================================================

/** @file test-integrator-api.c
* @author WBC, JKM
* @date July 2015
* @copyright Apache License, Version 2.0
* @brief Tests the RoadRunner integrator C API
**/

#include "rrc_api.h"
#include "rrc_logging_api.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char* raw_sbml[3] = {
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<!-- Created by XMLPrettyPrinter on 7/11/2014 from JDesigner 2.4.32 -->\r\n<sbml xmlns = \"http://www.sbml.org/sbml/level2\" level = \"2\" version = \"1\" xmlns:jd2 = \"http://www.sys-bio.org/sbml/jd2\">\r\n   <!--                     -->\r\n   <!--  Model Starts Here  -->\r\n   <!--                     -->\r\n   <model id = \"untitled\">\r\n      <listOfCompartments>\r\n         <compartment id = \"compartment\" size = \"1\"/>\r\n      </listOfCompartments>\r\n      <listOfSpecies>\r\n         <species id = \"Node0\" boundaryCondition = \"false\" initialConcentration = \"10\" compartment = \"compartment\"/>\r\n         <species id = \"Node1\" boundaryCondition = \"false\" initialConcentration = \"0\" compartment = \"compartment\"/>\r\n      </listOfSpecies>\r\n      <listOfParameters>\r\n         <parameter id = \"J0_k\" value = \"0.1\"/>\r\n      </listOfParameters>\r\n      <listOfReactions>\r\n         <reaction id = \"J0\" reversible = \"false\">\r\n            <listOfReactants>\r\n               <speciesReference species = \"Node0\" stoichiometry = \"1\"/>\r\n            </listOfReactants>\r\n            <listOfProducts>\r\n               <speciesReference species = \"Node1\" stoichiometry = \"1\"/>\r\n            </listOfProducts>\r\n            <kineticLaw>\r\n               <math xmlns = \"http://www.w3.org/1998/Math/MathML\">\r\n                  <apply>\r\n                     <times/>\r\n                     <ci>\r\n                           J0_k\r\n                     </ci>\r\n                     <ci>\r\n                           Node0\r\n                     </ci>\r\n                  </apply>\r\n               </math>\r\n            </kineticLaw>\r\n         </reaction>\r\n      </listOfReactions>\r\n   </model>\r\n</sbml>\r\n",
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<sbml xmlns=\"http://www.sbml.org/sbml/level2\" xmlns:jd2=\"http://www.sys-bio.org/sbml/jd2\" level=\"2\" version=\"1\">\r\n  <model id=\"feedback\" name=\"Feedback\">\r\n    <listOfCompartments>\r\n      <compartment id=\"compartment\" size=\"1\"/>\r\n    </listOfCompartments>\r\n    <listOfSpecies>\r\n      <species id=\"S1\" compartment=\"compartment\" initialConcentration=\"0\"/>\r\n      <species id=\"S2\" compartment=\"compartment\" initialConcentration=\"0\"/>\r\n      <species id=\"S3\" compartment=\"compartment\" initialConcentration=\"0\"/>\r\n      <species id=\"S4\" compartment=\"compartment\" initialConcentration=\"0\"/>\r\n      <species id=\"X0\" compartment=\"compartment\" initialConcentration=\"10\" boundaryCondition=\"true\"/>\r\n      <species id=\"X1\" compartment=\"compartment\" initialConcentration=\"0\" boundaryCondition=\"true\"/>\r\n    </listOfSpecies>\r\n    <listOfParameters>\r\n      <parameter id=\"J0_VM1\" value=\"10\"/>\r\n      <parameter id=\"J0_Keq1\" value=\"10\"/>\r\n      <parameter id=\"J0_h\" value=\"10\"/>\r\n      <parameter id=\"J4_V4\" value=\"2.5\"/>\r\n      <parameter id=\"J4_KS4\" value=\"0.5\"/>\r\n    </listOfParameters>\r\n    <listOfReactions>\r\n      <reaction id=\"J0\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"X0\"/>\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"S1\"/>\r\n        </listOfProducts>\r\n        <listOfModifiers>\r\n          <modifierSpeciesReference species=\"S4\"/>\r\n        </listOfModifiers>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <divide/>\r\n              <apply>\r\n                <times/>\r\n                <ci> J0_VM1 </ci>\r\n                <apply>\r\n                  <minus/>\r\n                  <ci> X0 </ci>\r\n                  <apply>\r\n                    <divide/>\r\n                    <ci> S1 </ci>\r\n                    <ci> J0_Keq1 </ci>\r\n                  </apply>\r\n                </apply>\r\n              </apply>\r\n              <apply>\r\n                <plus/>\r\n                <cn type=\"integer\"> 1 </cn>\r\n                <ci> X0 </ci>\r\n                <ci> S1 </ci>\r\n                <apply>\r\n                  <power/>\r\n                  <ci> S4 </ci>\r\n                  <ci> J0_h </ci>\r\n                </apply>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n      <reaction id=\"J1\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"S1\"/>\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"S2\"/>\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <divide/>\r\n              <apply>\r\n                <minus/>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 10 </cn>\r\n                  <ci> S1 </ci>\r\n                </apply>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 2 </cn>\r\n                  <ci> S2 </ci>\r\n                </apply>\r\n              </apply>\r\n              <apply>\r\n                <plus/>\r\n                <cn type=\"integer\"> 1 </cn>\r\n                <ci> S1 </ci>\r\n                <ci> S2 </ci>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n      <reaction id=\"J2\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"S2\"/>\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"S3\"/>\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <divide/>\r\n              <apply>\r\n                <minus/>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 10 </cn>\r\n                  <ci> S2 </ci>\r\n                </apply>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 2 </cn>\r\n                  <ci> S3 </ci>\r\n                </apply>\r\n              </apply>\r\n              <apply>\r\n                <plus/>\r\n                <cn type=\"integer\"> 1 </cn>\r\n                <ci> S2 </ci>\r\n                <ci> S3 </ci>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n      <reaction id=\"J3\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"S3\"/>\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"S4\"/>\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <divide/>\r\n              <apply>\r\n                <minus/>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 10 </cn>\r\n                  <ci> S3 </ci>\r\n                </apply>\r\n                <apply>\r\n                  <times/>\r\n                  <cn type=\"integer\"> 2 </cn>\r\n                  <ci> S4 </ci>\r\n                </apply>\r\n              </apply>\r\n              <apply>\r\n                <plus/>\r\n                <cn type=\"integer\"> 1 </cn>\r\n                <ci> S3 </ci>\r\n                <ci> S4 </ci>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n      <reaction id=\"J4\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"S4\"/>\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"X1\"/>\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <divide/>\r\n              <apply>\r\n                <times/>\r\n                <ci> J4_V4 </ci>\r\n                <ci> S4 </ci>\r\n              </apply>\r\n              <apply>\r\n                <plus/>\r\n                <ci> J4_KS4 </ci>\r\n                <ci> S4 </ci>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n    </listOfReactions>\r\n  </model>\r\n</sbml>\r\n;",
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<!-- Created by JarnacLite version 1.0.4965.360 on 2013-10-09 15:31 with libSBML version 5.8.0. -->\r\n<sbml xmlns=\"http://www.sbml.org/sbml/level2/version4\" level=\"2\" version=\"4\">\r\n  <model id=\"cell\" name=\"cell\">\r\n    <listOfCompartments>\r\n      <compartment id=\"compartment\" size=\"1\" />\r\n    </listOfCompartments>\r\n    <listOfSpecies>\r\n      <species id=\"Xo\" compartment=\"compartment\" initialConcentration=\"0\" boundaryCondition=\"true\" />\r\n      <species id=\"w\" compartment=\"compartment\" initialConcentration=\"0\" boundaryCondition=\"true\" />\r\n      <species id=\"x\" compartment=\"compartment\" initialConcentration=\"0.05\" boundaryCondition=\"false\" />\r\n    </listOfSpecies>\r\n    <listOfParameters>\r\n      <parameter id=\"k1\" value=\"0.9\" constant=\"true\" />\r\n      <parameter id=\"k2\" value=\"0.3\" constant=\"true\" />\r\n      <parameter id=\"k3\" value=\"0.7\" constant=\"true\" />\r\n    </listOfParameters>\r\n    <listOfReactions>\r\n      <reaction id=\"J1\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"Xo\" stoichiometry=\"1\" />\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"x\" stoichiometry=\"1\" />\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <plus />\r\n              <cn> 0.1 </cn>\r\n              <apply>\r\n                <divide />\r\n                <apply>\r\n                  <times />\r\n                  <ci> k1 </ci>\r\n                  <apply>\r\n                    <power />\r\n                    <ci> x </ci>\r\n                    <cn type=\"integer\"> 4 </cn>\r\n                  </apply>\r\n                </apply>\r\n                <apply>\r\n                  <plus />\r\n                  <ci> k2 </ci>\r\n                  <apply>\r\n                    <power />\r\n                    <ci> x </ci>\r\n                    <cn type=\"integer\"> 4 </cn>\r\n                  </apply>\r\n                </apply>\r\n              </apply>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n      <reaction id=\"_J1\" reversible=\"false\">\r\n        <listOfReactants>\r\n          <speciesReference species=\"x\" stoichiometry=\"1\" />\r\n        </listOfReactants>\r\n        <listOfProducts>\r\n          <speciesReference species=\"w\" stoichiometry=\"1\" />\r\n        </listOfProducts>\r\n        <kineticLaw>\r\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\r\n            <apply>\r\n              <times />\r\n              <ci> k3 </ci>\r\n              <ci> x </ci>\r\n            </apply>\r\n          </math>\r\n        </kineticLaw>\r\n      </reaction>\r\n    </listOfReactions>\r\n  </model>\r\n</sbml>"
};

static const char* sbml_desc[3] = {
    "decay",
    "feedback",
    "bistable"
};

int run_test_with_cvode(RRHandle _handle) {
    char *settingName, *settingDisplayName, *settingDesc, *settingHint;
    int settingType;
    int i;
    struct RRStringArray *strArray;

    // set active integrator to CVODE
    setCurrentIntegrator(_handle, "cvode");

    // Probe default (CVODE) integrator

    // Check that name == cvode
    {
        char* name = getCurrentIntegratorName(_handle);
        if (strcmp(name,"cvode")) {
            fprintf(stderr, "!! Expected integrator name to be 'cvode'\n");
            return 1;
        }
        freeText(name);
    }
    fprintf(stderr,"    Description: %s \n", getCurrentIntegratorDescription(_handle));
    fprintf(stderr,"    Hint: %s \n", getCurrentIntegratorHint(_handle));

    {
        int nparams = getNumberOfCurrentIntegratorParameters(_handle);
        fprintf(stderr, "    Number of parameters: %d\n", nparams);
        for (i = 0; i < nparams; ++i)
        {
            settingName = getCurrentIntegratorNthParameterName(_handle, i);
            settingDesc = getCurrentIntegratorNthParameterDescription(_handle, i);
            settingHint = getCurrentIntegratorNthParameterHint(_handle, i);
            settingDisplayName = getCurrentIntegratorNthParameterDisplayName(_handle, i);

            fprintf(stderr,"    %s\n", settingName);
            fprintf(stderr,"      Description: %s\n      Hint: %s\n      Display Name: %s\n", settingDesc, settingHint, settingDisplayName);

            freeText(settingName);
            freeText(settingDesc);
            freeText(settingDisplayName);
        }
    }


    fprintf(stderr,"\n  **** Test resetting parameters ****\n\n");

    fprintf(stderr,"    Set absolute tolerance to 1e-7\n");
    setCurrentIntegratorParameterDouble(_handle, "absolute_tolerance", 1e-7);
    fprintf(stderr,"    Result: %e\n", getCurrentIntegratorParameterDouble(_handle, "absolute_tolerance"));

    fprintf(stderr,"    Reset all integrator settings\n");
    resetCurrentIntegratorParameters(_handle);
    fprintf(stderr,"    Current value of absolute_tolerance: %e\n", getCurrentIntegratorParameterDouble(_handle, "absolute_tolerance"));


    fprintf(stderr,"\n  **** Test boolean parameters ****\n\n");

    fprintf(stderr,"    Initial value of multiple_steps: %d\n", getCurrentIntegratorParameterBoolean(_handle, "multiple_steps"));
    fprintf(stderr,"    Set multiple_steps to true\n");
    setCurrentIntegratorParameterBoolean(_handle, "multiple_steps", 1);

    fprintf(stderr,"    Current value of multiple_steps: %d\n", getCurrentIntegratorParameterBoolean(_handle, "multiple_steps"));



	fprintf(stderr, "\n  **** Test double vector parameters ****\n\n");
	//fprintf(stderr, getSBML(_handle));

	fprintf(stderr, "    Current value of absolute_tolerance: %e\n", getCurrentIntegratorParameterDouble(_handle, "absolute_tolerance"));
	fprintf(stderr, "    Set absolute_tolerance to [0.1, 0.2]\n");
	double a[2] = { 0.1, 0.2};
	setCurrentIntegratorParameterDoubleArray(_handle, "absolute_tolerance", a, 2);

	fprintf(stderr, "    Current value of absolute_tolerance: ");
	double* res = 0;
	int len = 0;
	getCurrentIntegratorParameterDoubleArray(_handle, "absolute_tolerance", &res, &len);
	for (int i = 0; i < len; ++i) {
		fprintf(stderr, "%e ", res[i]);
	}
	fprintf(stderr, "\n");
	fprintf(stderr, "\n");

	
	fprintf(stderr, "\n  **** Test setting individal tolerance ****\n\n");
	fprintf(stderr, "    Current value of absolute_tolerance: %e\n", getCurrentIntegratorParameterDouble(_handle, "absolute_tolerance"));
	fprintf(stderr, "    Set absolute tolerance for species \"node1\" to 0.3\n");
	setCurrentIntegratorIndividualTolerance(_handle, "Node1", 0.3);

	getCurrentIntegratorParameterDoubleArray(_handle, "absolute_tolerance", &res, &len);
	for (int i = 0; i < len; ++i) {
		fprintf(stderr, "%e ", res[i]);
	}
	fprintf(stderr, "\n");
	fprintf(stderr, "\n");

    fprintf(stderr,"    New list of parameters:\n");

    strArray = getListOfCurrentIntegratorParameterNames(_handle);
    for (i = 0; i < strArray->Count; ++i)
    {
        settingName = strArray->String[i];
        settingDesc = getCurrentIntegratorParameterDescription(_handle, settingName);
        settingHint = getCurrentIntegratorParameterHint(_handle, settingName);
        settingType = getCurrentIntegratorParameterType(_handle, settingName);

        fprintf(stderr,"    %s\n", settingName);
        fprintf(stderr,"    Type: %d\n    Description: %s\n    Hint: %s\n\n", settingType, settingDesc, settingHint);
    }

    fprintf(stderr,"\n  **** CVODE Simulation ****\n\n");

    // reset the model
    resetToOrigin(_handle);

    // Simulate
    {
        RRCDataPtr result;
        result = simulateEx(_handle, 0, 10, 11);
        fprintf(stderr,rrCDataToString(result));
        freeRRCData(result);
    }

    return 0;
}

int run_test_with_gillespie(RRHandle _handle) {
    char *settingName, *settingDesc, *settingHint;
    int settingType;
    int i;
    struct RRStringArray *strArray;

    fprintf(stderr,"\n  **** Test stochastic simulation ****\n\n");

    // reset the model
    resetToOrigin(_handle);

    // Add Gillespie Integrator to the mix and then grab updated info on all implemented integrators.
    setCurrentIntegrator(_handle, "gillespie");
    fprintf(stderr,"  Number of instantiated integrators:\t %d\n", getNumInstantiatedIntegrators(_handle));

    // Probe Gillespie integrator
    fprintf(stderr,"    %s \n", getCurrentIntegratorDescription(_handle));
    fprintf(stderr,"    %s \n", getCurrentIntegratorHint(_handle));
    fprintf(stderr,"    %d \n", getNumberOfCurrentIntegratorParameters(_handle));

    strArray = getListOfCurrentIntegratorParameterNames(_handle);
    for (i = 0; i < strArray->Count; ++i)
    {
        settingName = strArray->String[i];
        settingDesc = getCurrentIntegratorParameterDescription(_handle, settingName);
        settingHint = getCurrentIntegratorParameterHint(_handle, settingName);
        settingType = getCurrentIntegratorParameterType(_handle, settingName);

        fprintf(stderr,"    %s\n", settingName);
        fprintf(stderr,"    Type: %d\n    Description: %s\n    Hint: %s\n\n", settingType, settingDesc, settingHint);
    }

    fprintf(stderr, "Set Gillespie random seed to 12345\n");
    setCurrentIntegratorParameterUInt(_handle, "seed", 12345u);

    fprintf(stderr,"\n  **** Gillespie Simulation ****\n\n");

    // Simulate
    {
        RRCDataPtr result;
        result = simulateEx(_handle, 0, 10, 11);
        fprintf(stderr,rrCDataToString(result));
        freeRRCData(result);
    }

    return 0;
}

int run_test_with_rk4(RRHandle _handle) {
    char *settingName, *settingDesc, *settingHint;
    int settingType;
    int i;
    struct RRStringArray *strArray;

    // Simulate with RK4

    setCurrentIntegrator(_handle, "rk4");
    fprintf(stderr,"Number of instantiated integrators:\t %d\n", getNumInstantiatedIntegrators(_handle));

    // Probe Gillespie integrator
    fprintf(stderr,"    Description: %s \n", getCurrentIntegratorDescription(_handle));
    fprintf(stderr,"    Hint: %s \n", getCurrentIntegratorHint(_handle));
    fprintf(stderr,"    Parameters: %d \n", getNumberOfCurrentIntegratorParameters(_handle));

    strArray = getListOfCurrentIntegratorParameterNames(_handle);
    for (i = 0; i < strArray->Count; ++i)
    {
        settingName = strArray->String[i];
        settingDesc = getCurrentIntegratorParameterDescription(_handle, settingName);
        settingHint = getCurrentIntegratorParameterHint(_handle, settingName);
        settingType = getCurrentIntegratorParameterType(_handle, settingName);

        fprintf(stderr,"    %s\n", settingName);
        fprintf(stderr,"    Type: %d\n    Description: %s\n    Hint: %s\n\n", settingType, settingDesc, settingHint);
    }

    fprintf(stderr,"\n  **** RK4 Simulation ****\n\n");

    // reset the model
    resetToOrigin(_handle);

    // Simulate
    {
        RRCDataPtr result;
        result = simulateEx(_handle, 0, 10, 11);
        fprintf(stderr,rrCDataToString(result));
        freeRRCData(result);
    }

    return 0;
}

int run_test_with_rk45(RRHandle _handle) {
    char *settingName, *settingDesc, *settingHint;
    int settingType;
    int i;
    struct RRStringArray *strArray;

    // Simulate with RK4

    setCurrentIntegrator(_handle, "rk45");
    fprintf(stderr,"Number of instantiated integrators:\t %d\n", getNumInstantiatedIntegrators(_handle));

    // Probe Gillespie integrator
    fprintf(stderr,"    Description: %s \n", getCurrentIntegratorDescription(_handle));
    fprintf(stderr,"    Hint: %s \n", getCurrentIntegratorHint(_handle));
    fprintf(stderr,"    Parameters: %d \n", getNumberOfCurrentIntegratorParameters(_handle));

    strArray = getListOfCurrentIntegratorParameterNames(_handle);
    for (i = 0; i < strArray->Count; ++i)
    {
        settingName = strArray->String[i];
        settingDesc = getCurrentIntegratorParameterDescription(_handle, settingName);
        settingHint = getCurrentIntegratorParameterHint(_handle, settingName);
        settingType = getCurrentIntegratorParameterType(_handle, settingName);

        fprintf(stderr,"    %s\n", settingName);
        fprintf(stderr,"    Type: %d\n    Description: %s\n    Hint: %s\n\n", settingType, settingDesc, settingHint);
    }

    fprintf(stderr,"\n  **** RK45 Simulation ****\n\n");

    // reset the model
    resetToOrigin(_handle);

    // Simulate
    {
        RRCDataPtr result;
        result = simulateEx(_handle, 0, 10, 11);
        fprintf(stderr,rrCDataToString(result));
        freeRRCData(result);
    }

    return 0;
}

//call with arguments, -m"modelFilePath" -r"resultFileFolder" -t"TempFolder" -s"Suites"
int main(int argc, char* argv[])
{
    int n;
    // enable logging
//     setLogLevel("debug");
    enableLoggingToFile();
    {
        char* t = getLogFileName();
        fprintf(stderr,"Enabling logging to %s\n", t);
        freeText(t);
    }

    // print version
    {
        char* t = getVersionStr();
        fprintf(stderr,"RoadRunner version %s\n", t);
        freeText(t);
    }

    fprintf(stderr,"Initializing RoadRunner...\n");
    RRHandle _handle = createRRInstance();

    {
        char* t = getCurrentIntegratorName(_handle);
        fprintf(stderr, "  Default integrator is %s\n", t);
        freeText(t);
    }

    // set integrator to gillespie before loading a model to test for crashing
    setCurrentIntegrator(_handle, "gillespie");

    for(n = 0; n<(sizeof(raw_sbml)/sizeof(raw_sbml[0])); ++n) {
        struct RRStringArray *strArray;
        char *settingName, *settingDesc, *settingHint;
        char *_intgList;
        int settingType;
        int k;

        // load the model
        {
            const char* modelstr = raw_sbml[n];
            const char* modeldsc = sbml_desc[n];
            fprintf(stderr,"\n  **** Loading model '%s'\n\n", modeldsc);
            loadSBML(_handle, modelstr);
        }


        fprintf(stderr,"\n  ****\n\n");


        // Grab info on all implemented integrators.
        fprintf(stderr,"  Number of registered integrators:\t %d\n", getNumRegisteredIntegrators());
        for (k = 0; k < getNumRegisteredIntegrators(); ++k) {
            fprintf(stderr,"    %d:\n", k);
            fprintf(stderr,"     Name: %s\n", getRegisteredIntegratorName(k));
            fprintf(stderr,"     Hint: %s\n", getRegisteredIntegratorHint(k));
            fprintf(stderr,"     Desc: %s\n", getRegisteredIntegratorDescription(k));
        }


        fprintf(stderr,"\n  **** Probe instantiated integrators\n\n");

        fprintf(stderr,"  Number of instantiated integrators:\t %d\n", getNumInstantiatedIntegrators(_handle));

        // instantiate all integrators to test for problems
        setCurrentIntegrator(_handle, "cvode");
        setCurrentIntegrator(_handle, "rk4");
        setCurrentIntegrator(_handle, "euler");
        setCurrentIntegrator(_handle, "gillespie");
        setCurrentIntegrator(_handle, "cvode");

        if(run_test_with_rk4(_handle))
            return 1;

        if(run_test_with_cvode(_handle))
            return 1;

        if(run_test_with_gillespie(_handle))
            return 1;

        if(run_test_with_rk45(_handle))
            return 1;
    }

    freeRRInstance(_handle);
	
    return 0;
}




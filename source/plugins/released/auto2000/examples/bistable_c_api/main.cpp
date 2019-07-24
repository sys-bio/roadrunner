#pragma hdrstop
#include <iostream>
#include <string>
#include "rrc_api.h"
#include "telplugins_c_api.h"

using namespace std;
//using namespace rrc;
//using namespace tlp; //Plugin stuff
//using namespace autoplugin;

int main()
{
    string tempFolder("r:/temp");
    string sbmlFile("../models/bistable.xml");

    rrc::RRHandle  rri = NULL;
    try
    {
        string autoPluginName("rrp_auto2000");
        string autoOutputParserPluginName("rrp_AutoOutputParser");
        tpSetLogLevel("LOG_DEBUG");

        cout<<"Current log level is: "<<tpGetLogLevel();
        //A roadrunner object
        rri = rrc::createRRInstance();
        rrc::setTempFolder(rri, tempFolder.c_str());

        //A Plugin manager, using a roadrunner instance
        TELHandle pm = tpCreatePluginManager("");

        //Load auto plugin
        if(!tpLoadPlugin(pm, autoPluginName.c_str()))
        {
            tpLogMsg(clError, "Failed to load auto plugin");
            return 0;
        }

        //Get a plugin handle
        TELHandle autoPlugin = tpGetPlugin(pm, autoPluginName.c_str());
        if(!autoPlugin)
        {
            tpLogMsg(clError, "Problem..");
            throw("AutoPlugin don't exist");
        }

        //A serious client would check if these calls are succesfull or not
        tpSetPluginProperty(autoPlugin, "ScanDirection", "Negative");
        tpSetPluginProperty(autoPlugin, "PrincipalContinuationParameter", "k3");
        tpSetPluginProperty(autoPlugin, "PCPLowerBound", "0.2");
        tpSetPluginProperty(autoPlugin, "PCPUpperBound", "1.2");

        //We can set the sbml parameter here or load into the roadrunner instance first
        //autoPlugin->setParameter("SBML", getFileContent(sbmlFile).c_str());

        //The load function does throw if file is not found.. is that what we want??
        if(!rrc::loadSBML(rri, sbmlFile.c_str()))
        {
            throw("Failed loading sbml model");
        }

        bool runInThread = true;
        if(!tpExecutePluginEx(autoPlugin, runInThread))
        {
            tpLogMsg(clError, "Problem executing the Auto plugin");
            return 0;
        }

        if(runInThread)
        {
            while(tpIsPluginWorking(autoPlugin))
            {
                tpLogMsg(clInfo, "Plugin is working..");
            }
        }

        tpLogMsg(clInfo, "Auto plugin is done.");
        TELHandle bifurcationDiagram = tpGetPluginProperty(autoPlugin, "BifurcationDiagram");
        if(bifurcationDiagram)
        {
            string *temp =  (string*) tpGetPropertyValueHandle(bifurcationDiagram);
            tpLogMsg(clInfo, temp->c_str());
        }

        //Parse auto output
        if(!tpLoadPlugin(pm, autoOutputParserPluginName.c_str()))
        {
            tpLogMsg(clError, "Failed to load auto output plugin");
            return 0;
        }

        //Get a plugin handle
        TELHandle parseAutoOutput = tpGetPlugin(pm, autoOutputParserPluginName.c_str());
        if(!parseAutoOutput)
        {
            tpLogMsg(clError, "Problem..");
            throw("The parseAutoOutput Plugin don't exist");
        }

        tpExecutePluginEx(parseAutoOutput, false);

        //Check if there were special solution points
        TELHandle solutionPoints = tpGetPluginProperty(parseAutoOutput, "LabeledSolutionPoints");

        char* msg = tpGetPropertyValueAsString(solutionPoints);

        tpLogMsg(clInfo, msg);
        tpFreeText(msg);
        tpUnLoadPlugin(pm, parseAutoOutput);
        tpUnLoadPlugin(pm, autoPlugin);
    }
    catch(...)
    {
        tpLogMsg(clError, "Holy cow problem...!");
    }

    return 0;
}

#if defined(CG_UI)

#if defined(STATIC_APP)
    #pragma comment(lib, "roadrunner-static.lib")
    #pragma comment(lib, "telplugins-static.lib")
    #pragma comment(lib, "roadrunner_c_api-static.lib")
    #pragma comment(lib, "telplugins_c_api-static.lib")
    #pragma comment(lib, "libsbml-static.lib")
    #pragma comment(lib, "libxml2_xe.lib")
#else
    #pragma comment(lib, "roadrunner_c_api.lib")
    #pragma comment(lib, "telplugins_c_api.lib")
#endif

#pragma comment(lib, "poco_foundation-static.lib")
//#pragma comment(lib, "cg32.lib")
#endif


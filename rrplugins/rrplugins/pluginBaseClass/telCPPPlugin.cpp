#pragma hdrstop
#include "telCPPPlugin.h"
//---------------------------------------------------------------------------

namespace tlp
{

CPPPlugin::CPPPlugin(const string& name, const string& cat)//, const PluginManager* pm) rrc::THostInterface* aRR,const PluginManager* pm)
:
Plugin(name, cat, "CPP")//, pm)
{}

CPPPlugin::~CPPPlugin()
{}

string CPPPlugin::getImplementationLanguage()
{
    return "CPP";
}

/*
void CPPPlugin::setHostInterface(rrc::THostInterface* _hostInterface) {
    mhostInterface = _hostInterface;
}*/

}


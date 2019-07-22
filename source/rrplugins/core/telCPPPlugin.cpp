#pragma hdrstop
#include "telCPPPlugin.h"
//---------------------------------------------------------------------------

namespace rr
{

CPPPlugin::CPPPlugin(const string& name, const string& cat, RoadRunner* aRR, const PluginManager* pm)
:
Plugin(name, cat, aRR, "CPP", pm)
{}

CPPPlugin::~CPPPlugin()
{}

string CPPPlugin::getImplementationLanguage()
{
    return "CPP";
}

}


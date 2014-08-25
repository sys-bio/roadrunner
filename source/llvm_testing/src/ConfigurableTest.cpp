/*
 * ConfigurableTest.cpp
 *
 *  Created on: Sep 25, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ConfigurableTest.h"
#include "rrRoadRunner.h"
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;

namespace rr
{

ConfigurableTest::ConfigurableTest()
{
    // TODO Auto-generated constructor stub

}

ConfigurableTest::~ConfigurableTest()
{
    // TODO Auto-generated destructor stub
}

_xmlNode* ConfigurableTest::createConfigNode()
{
    return 0;
}

void ConfigurableTest::loadConfig(const _xmlDoc* doc)
{
    std::string str;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p1");
    cout << "p1 value: " << str << endl;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p2");
    cout << "p2 value: " << str << endl;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p3");
    cout << "p3 value: " << str << endl;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p4");
    cout << "p4 value: " << str << endl;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p5");
    cout << "p5 value: " << str << endl;

    str = Configurable::getParameterStringValue(doc, "cap_name", "p6");
    cout << "p6 value: " << str << endl;

}

bool ConfigurableTest::test()
{

    _xmlNode *p1 = Configurable::createParameterNode("p1", "p1_hint", "p1_value");
    _xmlNode *p2 = Configurable::createParameterNode("p2", "p2_hint", "p2_value");
    _xmlNode *p3 = Configurable::createParameterNode("p3", "p3_hint", "p3_value");
    _xmlNode *p4 = Configurable::createParameterNode("p4", "p4_hint", "p4_value");
    _xmlNode *p5 = Configurable::createParameterNode("p5", "p5_hint", 12);
    _xmlNode *p6 = Configurable::createParameterNode("p6", "p6_hint", 3.14);

    _xmlNode *cap = Configurable::createCapabilityNode("cap_name", "cap_method", "cap_desc");

    Configurable::addChild(cap, p1);
    Configurable::addChild(cap, p2);
    Configurable::addChild(cap, p3);
    Configurable::addChild(cap, p4);
    Configurable::addChild(cap, p5);
    Configurable::addChild(cap, p6);


    _xmlNode *caps = Configurable::createCapabilitiesNode("caps_name", "caps_desc");

    Configurable::addChild(caps, cap);

    std::string str = Configurable::xmlFromConfigNode(caps);

    Configurable::loadXmlConfig(str, this);



    return true;
}



void ConfigurableTest::testRoadRunner(const std::string& sbml, const std::string configFname)
{
    ifstream inFile(configFname.c_str());


    stringstream strStream;
    strStream << inFile.rdbuf();//read the file
    string str = strStream.str();//str holds the content of the file


    RoadRunner r;

    std::string xml = r.getConfigurationXML();

    cout << "original xml: " << xml << endl;

    r.load(sbml);

    xml = r.getConfigurationXML();

    cout << "loaded sbml xml: " << xml << endl;

    r.setConfigurationXML(str);

    xml = r.getConfigurationXML();

    cout << "after set xml: " << xml << endl;

}

} /* namespace rrllvm */

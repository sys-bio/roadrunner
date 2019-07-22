#pragma hdrstop
#include <sstream>
#include <iomanip>
#include "../../rrUtils.h"
#include "telPlugin.h"
#include "../common/telProperty.h"
#include "../common/telTelluriumData.h"
#include <stdexcept>

//---------------------------------------------------------------------------

namespace rr
{
using namespace std;

Plugin::Plugin( const string&           name,
                const string&           category,
                RoadRunner*             aRR,
                const string&           language,
                const PluginManager*    pm)
:
mName(name),
mAuthor("Totte Karlsson, J Kyle Medley and Wilbert Copeland"),
mCategory(category),
mDescription("<none>"),
mHint("<none>"),
mVersion("0.0"),
mCopyright("Totte Karlsson, J Kyle Medley, Wilbert Copeland and Herbert Sauro, Systems Biology, UW 2012-2015"),
mRR(aRR),
mWorkStartedEvent(NULL),
mWorkProgressEvent(NULL),
mWorkFinishedEvent(NULL),
mProperties(),//name, category),
mImplementationLanguage(language),
mPM(pm),
mTerminate(false),
mIsWorking(false)
{}

Plugin::~Plugin()
{}

const PluginManager* Plugin::getPluginManager()
{
    return mPM;
}

void Plugin::terminate()
{
    if(!mIsWorking)
    {
        Log(lWarning) << "Can't terminate a non working plugin..";
        return;
    }
    mTerminate = true;
}

bool Plugin::isBeingTerminated() const
{
    return mTerminate;
}

bool Plugin::wasTerminated() const
{
    if(mTerminate)
    {
        if(mIsWorking == false)
        {
            return true;
        }
        else
        {
            //Still working...
            return false;
        }
    }
    return false;
}

bool Plugin::hasStartedEvent() const
{
    return (mWorkStartedEvent) ? true : false;
}

bool Plugin::hasProgressEvent() const
{
    return (mWorkProgressEvent) ? true : false;
}

bool Plugin::hasFinishedEvent() const
{
    return (mWorkFinishedEvent) ? true : false;
}

void Plugin::WorkStartedEvent(void* data1, void* data2)
{
    if(mWorkStartedEvent)
    {
        mWorkStartedEvent(data1, data2);
    }
}

void Plugin::WorkProgressEvent(void* data1, void* data2)
{
    if(mWorkProgressEvent)
    {
        mWorkProgressEvent(data1, data2);
    }
}

void Plugin::WorkFinishedEvent(void* data1, void* data2)
{
    if(mWorkFinishedEvent)
    {
        mWorkFinishedEvent(data1, data2);
    }
}

bool Plugin::assignRoadRunnerInstance(RoadRunner* rr)
{
    mRR = rr;
    return mRR ? true : false;
}

RoadRunner* Plugin::getRoadRunnerInstance()
{
    return mRR;
}

bool Plugin::resetPlugin()
{
    //Do whats needed in descendants
    return true;
}

/*string Plugin::getPluginPropertiesAsXML()
{
    return mProperties.asXML();
}*/

pair<void*, void*> Plugin::getWorkStartedData()
{
    return pair<void*, void*>(mWorkStartedData1, mWorkStartedData2);
}

pair<void*, void*> Plugin::getWorkProgressData()
{
    return pair<void*, void*>(mWorkProgressData1, mWorkProgressData2);
}

pair<void*, void*> Plugin::getWorkFinishedData()
{
    return pair<void*, void*>(mWorkFinishedData1, mWorkFinishedData2);
}

bool Plugin::assignOnStartedEvent(PluginEvent startedFnc, void* userData1, void* userData2)
{
    mWorkStartedEvent = startedFnc;
    mWorkStartedData1 = userData1;
    mWorkStartedData2 = userData2;
    return true;
}

bool Plugin::assignOnProgressEvent(PluginEvent progress, void* userData1, void* userData2)
{
    mWorkProgressEvent = progress;
    mWorkProgressData1 = userData1;
    mWorkProgressData2 = userData2;
    return true;
}

bool Plugin::assignOnFinishedEvent(PluginEvent endFnc, void* userData1, void* userData2)
{
    mWorkFinishedEvent = endFnc;
    mWorkFinishedData1 = userData1;
    mWorkFinishedData2 = userData2;
    return true;
}

bool Plugin::isWorking() const
{
    return false;
}

void Plugin::setLibraryName(const string& libName)
{
    mLibraryName = libName;
}

void Plugin::setPropertyByString(const string& nameOf, const char* value)
{
    string val(value);
    if(!mProperties.setProperty(nameOf, val))
    {
        stringstream msg;
        msg<<"Failed setting property value; No property with name: "<<nameOf;
        throw(Exception(msg.str()));
    }
}

void Plugin::setPropertyValue(const string& nameOf, const void* value)
{
    PropertyBase* property = mProperties.getProperty(nameOf);
    if(property)
    {
        string type = property->getType();
        if(type == "bool")
        {
            Property<bool>* prop = dynamic_cast< Property<bool>* >(property);
            const bool* theData = (bool*) value;
            prop->setValue(*theData);
            return;
        }

        if(type == "int")
        {
            Property<int>* prop = dynamic_cast< Property<int>* >(property);
            const int* theData = (int*) value;
            prop->setValue(*theData);
            return;
        }

        if(type == "double")
        {
            Property<double>* prop = dynamic_cast< Property<double>* >(property);
            if (!prop)
                throw std::runtime_error("Plugin::setPropertyValue: failed to cast to double property");
            const double* theData = (double*) value;
            prop->setValue(*theData);
            return;
        }

        if(type == "std::string")
        {
            Property<string>* prop = dynamic_cast< Property<string>* >(property);
            const string* theData = (string*) value;
            prop->setValue(*theData);
            return;
        }

        if(type == "telluriumData")
        {
            Property<TelluriumData>* prop = dynamic_cast< Property<TelluriumData>* >(property);
            const TelluriumData* theData = (TelluriumData*) value;
            prop->setValue(*theData);
            return;
        }

        if(type == "listOfProperties")
        {
            Property<Properties>* props = dynamic_cast< Property<Properties>* >(property);
            const Properties* theData = (Properties*) value;
            props->setValue(*theData);
            return;
        }

        stringstream msg;
        msg<<"Conversion of property of type: \""<<type<<" is not implemented";
        throw(Exception(msg.str()));
    }

    stringstream msg;
    msg<<"Failed setting property value; No property with name: "<<nameOf;
    throw(Exception(msg.str()));
}

string Plugin::getName()
{
    return mName;
}

string Plugin::getLibraryName()
{
    return mLibraryName;
}

string Plugin::getAuthor()
{
    return mAuthor;
}

string Plugin::getCategory()
{
    return mCategory;
}

string Plugin::getDescription()
{
    return mDescription;
}

string Plugin::getHint()
{
    return mHint;
}

string Plugin::getVersion()
{
    return mVersion;
}

string Plugin::getCopyright()
{
    return mCopyright;
}

//Overload in descendants and add info there.
string Plugin::getStatus()
{
    stringstream msg;
    msg<<"The following is a status message for Plugin: "<<mName<<"\n";
    return msg.str();
}

//Obs. subclasses may over ride this function and add more info
string Plugin::getInfo()
{
    stringstream msg;
    msg<<setfill('.');
    msg<<setw(30)<<left<<"Name"<<mName<<"\n";
    msg<<setw(30)<<left<<"Author"<<mAuthor<<"\n";
    msg<<setw(30)<<left<<"Category"<<mCategory<<"\n";
    msg<<setw(30)<<left<<"Description"<<mDescription<<"\n";
    msg<<setw(30)<<left<<"Hint"<<mHint<<"\n";
    msg<<setw(30)<<left<<"Version"<<mVersion<<"\n";
    msg<<setw(30)<<left<<"Copyright"<<mCopyright<<"\n";
    return msg.str();
}

string Plugin::getExtendedInfo()
{
    stringstream msg;
    msg<<getInfo();
    msg<<"\nPlugin Properties Info\n";
    Properties* props = getProperties();
    msg<<(*props);
    return msg.str();
}

unsigned int Plugin::getPDFManualByteSize()
{
    return 0;
}

unsigned char* Plugin::getManualAsPDF() const
{
    return NULL;
}

rr::StringList Plugin::getPropertyNames()
{
    return mProperties.getNames();
}

Properties* Plugin::getProperties()
{
    return &mProperties;
}

PropertyBase* Plugin::getProperty(const string& propName)
{
    return mProperties.getProperty(propName);
}

string Plugin::getPropertyValueAsString(const string& propName)
{
    PropertyBase* prop =  mProperties.getProperty(propName);
    if(prop)
    {
        return prop->getValueAsString();
    }
    else
    {
        stringstream str;
        str<<"No property with name: "<<propName;
        throw(Exception(str.str()));
    }
}

void* Plugin::getPropertyValueHandle(const string& propName)
{
    PropertyBase* prop =  mProperties.getProperty(propName);
    if(prop)
    {
        return prop->getValueHandle();
    }
    else
    {
        stringstream str;
        str<<"No property with name: "<<propName;
        throw(Exception(str.str()));
    }
}

string Plugin::getResult()
{
    return "This plugin don't have any result..";
}

}

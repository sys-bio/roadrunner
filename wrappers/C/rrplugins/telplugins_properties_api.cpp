#pragma hdrstop
#include <string>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrOSSpecifics.h"
#include "rrplugins/common/telTelluriumData.h"
#include "telplugins_properties_api.h"
#include "telplugins_c_api.h"
#include "telplugins_cpp_support.h"
#include "rrplugins/common/telProperty.h"
#include "rrplugins/common/telPropertyBase.h"

#include <iostream>
//---------------------------------------------------------------------------

using namespace std;
using namespace rr;
using namespace tlpc;
//using tlp::TelluriumData;

TELHandle tlp_cc tpCreateProperty(const char* label, const char* type, const char* _hint, void* value)
{
    start_try
        char* hint = (char*) _hint;
        if(!hint)
        {
            hint = (char*) "";
        }

        if(string(type) == string("bool"))
        {
            bool iniVal  = false;
            if(value != NULL)
            {
                //cast it
                bool* val = (bool*) value;
                iniVal = (*val);
            }
            Property<bool>* para = new Property<bool>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        if(string(type) == string("int"))
        {
            int iniVal  = 0;
            if(value != NULL)
            {
                //cast it
                int* val = (int*) value;
                iniVal = (*val);
            }
            Property<int> *para = new Property<int>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        //Don't support type 'float', it just causes problems. Make it a double
        if(string(type) == string("float"))
        {
            double iniVal  = 0;
            if(value != NULL)
            {
                //cast it
                double* dVal = (double*) value;
                iniVal = (*dVal);
            }
            Property<double> *para = new Property<double>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        if(string(type) == string("double"))
        {
            double iniVal  = 0;
            if(value != NULL)
            {
                //cast it
                double* dVal = (double*) value;
                iniVal = (*dVal);
            }
            Property<double> *para = new Property<double>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        //if(string(type) == string("string"))
        //{
        //    char* iniVal  = "";
        //    if(value != NULL)
        //    {
        //        //cast it
        //        char* *val = (char**) value;
        //        iniVal = (*val);
        //    }
        //    Property<char*> *para = new Property<char*>(iniVal, label, hint);
        //    return para;
        //}

        if(string(type) == string("std::string") || string(type) == string("string"))
        {
            string iniVal  = "";
            if(value != NULL)
            {
                //cast it
                string* val = (string*) value;
                iniVal = (*val);
            }
            Property<string> *para = new Property<string>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        if(string(type) == string("listOfProperties"))
        {
            Properties iniVal;
            if(value != NULL)
            {
                //cast it
                Properties* val = (Properties*) value;
                iniVal = (*val);
            }

            Property<Properties> *para = new Property<Properties>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        if(string(type) == string("telluriumData"))
        {
            TelluriumData iniVal;
            if(value != NULL)
            {
                //cast it
                TelluriumData* val = (TelluriumData*) value;
                iniVal = (*val);
            }

            Property<TelluriumData> *para = new Property<TelluriumData>(iniVal, label, hint);
            gHM.registerHandle(para, typeid(static_cast<PropertyBase*>(para)).name());
            return para;
        }

        return NULL;
    catch_ptr_macro
}

bool tlp_cc tpFreeProperty(TELHandle paraHandle)
{
    start_try
        PropertyBase* para   = castHandle<PropertyBase>(paraHandle, __FUNC__);
        delete para;
        return true;
    catch_bool_macro
}

TELHandle tlp_cc tpCreatePropertyList()
{
    start_try
        Properties* props   = new Properties();
        gHM.registerHandle(props, typeid(props).name());
        if(!props)
        {
            throw("Failed to create a list of Properties");
        }
        else
        {
            return props;
        }
    catch_ptr_macro
}

bool tlp_cc tpFreeProperties(TELHandle handle)
{
    start_try
        Properties* props   = castHandle<Properties>(handle, __FUNC__);
        delete props;
        return true;
    catch_bool_macro
}

bool tlp_cc tpAddPropertyToList(TELHandle handle, TELHandle para)
{
    start_try
        Properties* paras   = castHandle<Properties>(handle, __FUNC__);
        PropertyBase* bPara = castHandle<PropertyBase>(para, __FUNC__);
        paras->add(bPara, false);
        return true;
    catch_bool_macro
}

char* tlp_cc tpGetPropertyInfo(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        stringstream s;
        s<<"Name="<<para->getName()<<"\tType="<<para->getType()<<"\tDescription="<<para->getDescription()<<"\tHint="<<para->getHint();
        return rr::createText(s.str());
    catch_ptr_macro
}

bool tlp_cc tpSetPropertyDescription(TELHandle handle, const char* value)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        para->setDescription(string(value));
        return true;
    catch_bool_macro
}

char* tlp_cc tpGetPropertyDescription(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        return rr::createText(para->getDescription());
    catch_ptr_macro
}

bool tlp_cc tpSetPropertyHint(TELHandle handle, const char* value)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        para->setHint(string(value));
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetPropertyByString(TELHandle handle, const char* value)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        if(value)
        {
            string val(value);
            para->setValueFromString(val);
            return true;
        }
        return false;
    catch_bool_macro
}

bool tlp_cc tpSetBoolProperty(TELHandle handle, bool value)
{
    start_try
        Property<bool>* para = castHandle< Property<bool> >(handle, __FUNC__);
        para->setValue(value);
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetBoolProperty(TELHandle handle, bool* value)
{
    start_try
        Property<bool>* para = castHandle< Property<bool> >(handle, __FUNC__);
        (*value) = para->getValue();
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetIntProperty(TELHandle handle, int value)
{
    start_try
        Property<int>* para = castHandle< Property<int> >(handle,__FUNC__);
        para->setValue(value);
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetIntProperty(TELHandle handle, int *value)
{
    start_try
        Property<int>* para = castHandle< Property<int> >(handle, __FUNC__);
        (*value) = para->getValue();
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetDoubleProperty(TELHandle handle, double value)
{
    start_try
        Property<double>* para = castHandle< Property<double> >(handle, __FUNC__);
        para->setValue(value);
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetDoubleProperty(TELHandle handle, double *value)
{
    start_try
        Property<double>* para = castHandle< Property<double> >(handle, __FUNC__);
        (*value) = para->getValue();
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetStringProperty(TELHandle handle, char* value)
{
    start_try
        PropertyBase* base = castHandle< PropertyBase >(handle, __FUNC__);
        if(!base) {
            Log(lError) << "tpSetStringProperty: Failed to get handle";
            throw std::runtime_error("Failed to get handle");
        }
        Property<string>* para = dynamic_cast< Property<string>* >(base);
        if(!para) {
            // can't dyncast string props; check typeid, better than nothing
            //if (typeid(*base).name().find("Property") == std::string::npos) {
            //Log(lError) << "tpSetStringProperty: Failed to cast property";
            //Log(lError) << "tpSetStringProperty: typeinfo = " << typeid(*base).name();
            //}
            para = static_cast< Property<string>* >(base); // so much for type safety
            //throw std::runtime_error("Failed to cast property");
        }
        string temp(value);
        para->setValueFromString(temp);
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetStringProperty(TELHandle handle, const char* (*value))
{
    start_try
        Property<string>* para = castHandle< Property<string> >(handle, __FUNC__);

        (*value) = para->getValue().c_str();
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetListProperty(TELHandle handle, void* value)
{
    start_try
        Property<Properties>* para = castHandle< Property<Properties> >(handle, __FUNC__);
        para->setValue((Properties*)(value));
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetListProperty(TELHandle handle, void* (value))
{
    start_try
        Property<Properties>* para = castHandle< Property<Properties> >(handle, __FUNC__);
        Properties* assignTo = castHandle<Properties>(value, __FUNC__);

        (assignTo) = (para->getValuePointer());
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataProperty(TELHandle handle, void* value)
{
    start_try
        PropertyBase* base = castHandle< PropertyBase >(handle, __FUNC__);
        if(!base) {
            Log(lError) << "tpSetTelluriumDataProperty: Failed to get handle";
            throw std::runtime_error("Failed to get handle");
        }
        Property<TelluriumData>* para = dynamic_cast< Property<TelluriumData>* >(base);
        if(!para) {
            Log(lError) << "tpSetTelluriumDataProperty: Failed to cast property";
            throw std::runtime_error("Failed to cast property");
        }
        TelluriumData* data = (TelluriumData*) value;
        data->byteCheck();
        para->setValue(*data);
        return true;
    catch_bool_macro
}

bool tlp_cc tpGetTelluriumDataProperty(TELHandle handle, void* value)
{
    start_try
        Property<TelluriumData>* para = castHandle< Property<TelluriumData> >(handle, __FUNC__);
        TelluriumData* assignTo = castHandle< TelluriumData >(value, __FUNC__);
        (*assignTo) = (para->getValueReference());
        return true;
    catch_bool_macro
}

char* tlp_cc tpGetPropertyValueAsString(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        string val = para->getValueAsString();
        return rr::createText(val);
    catch_ptr_macro
}

void* tlp_cc tpGetPropertyValueHandle(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        return para->getValueHandle();
    catch_ptr_macro
}

char* tlp_cc tpGetPropertyName(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        return rr::createText(para->getName());
    catch_ptr_macro
}

char* tlp_cc tpGetPropertyHint(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        return rr::createText(para->getHint());
    catch_ptr_macro
}

char* tlp_cc tpGetPropertyType(TELHandle handle)
{
    start_try
        PropertyBase* para = castHandle<PropertyBase>(handle, __FUNC__);
        char* text = rr::createText(para->getType());
        return text;
    catch_ptr_macro
}

TELHandle tlp_cc tpGetFirstProperty(TELHandle handle)
{
    start_try
        Properties *paras = castHandle<Properties>(handle, __FUNC__);
        return paras->getFirst();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetNextProperty(TELHandle handle)
{
    start_try
        Properties *paras = castHandle<Properties>(handle, __FUNC__);
        return paras->getNext();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetPreviousProperty(TELHandle handle)
{
    start_try
        Properties *paras = castHandle<Properties>(handle, __FUNC__);
        return paras->getPrevious();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetCurrentProperty(TELHandle handle)
{
    start_try
        Properties *paras = castHandle<Properties>(handle, __FUNC__);
        return paras->getCurrent();
    catch_ptr_macro
}

bool tlp_cc tpClearPropertyList(TELHandle handle)
{
    start_try
        Properties* paras = castHandle<Properties>(handle, __FUNC__);
        return paras->clear();
    catch_bool_macro
}

TELHandle tlp_cc tpGetProperty(TELHandle handle, const char* name)
{
    start_try
        Properties* props = castHandle<Properties>(handle, __FUNC__);
        return props->getProperty(name);
    catch_ptr_macro
}

char* tlp_cc tpGetNamesFromPropertyList(TELHandle handle)
{
    start_try
        Properties* paras = castHandle<Properties>(handle, __FUNC__);
        rr::StringList aList;
        for(int i = 0; i < paras->count(); i++)
        {
            aList.add((*paras)[i]->getName());
        }
        return rr::createText(aList.AsString().c_str());

    catch_ptr_macro
}

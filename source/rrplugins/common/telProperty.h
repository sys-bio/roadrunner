/**
 * @file telProperty.h
 * @brief Property implementation
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/
#ifndef telPropertyH
#define telPropertyH
#include <vector>
#include <string>
#include <sstream>
#include "../../rrException.h"
#include "../../rrLogger.h"
#include "../../rrStringUtils.h"
#include "telPropertyBase.h"
#include "telProperties.h"
#include "../../rrUtils.h"
#include "../../c/rrStringList.h"
#include "../../rrConstants.h"
//#include "telTelluriumData.h"
#include "rr-libstruct\lsExporter.h"
#include "rr-libstruct\lsLA.h"
#include "rr-libstruct\lsLibla.h"

//---------------------------------------------------------------------------
namespace rr
{
    using rr::gNoneString;
    using std::string;
    using std::stringstream;
/**
    Template function that returns the name of a particular type, e.g. a value of type int, returns a string "int".
    Each type need to be specialized.
*/
template <class T>
string getPropertyType(const T& val);

/**
    \brief Template class implementing a PluginProperty.

    The characteristics of a Plugin Property is its type and its value. Various functions assist in setting/getting
    the properties value. Its base class, PropertyBase, encapsulate the properties name, hint and description.

    Of importance is the ability to get a pointer to its internal value, e.g. \a getValuePointer(), or \a getValueHandle(), as it can be used as a
    handle in the derived Plugins C API.
*/
template<class T>
class Property : public PropertyBase
{
    public:
                                        /**
                                            Property constructor. Creates a property, assigning a value, name and optionally a hint.
                                        */
                                        Property(   const T& value = T(),                const string& name = gNoneString,
                                                    const string& hint = gNoneString,    const string& descr = gNoneString,
                                                    const string& alias = gEmptyString,   bool readonly = false);

                                        /**
                                            Property copy constructor. Creates a property, from another property.
                                        */
                                        Property(const Property<T>& para);

                                        /**
                                            Property destructor. Deallocate any memory allocated by the property.
                                        */
                                       ~Property();

                                        /**
                                            Property assignment operator. Deep copy of the property on the right side of the assignment.
                                        */
        Property<T>&                    operator=(const Property<T>& rhs);

                                        /**
                                            Set a property value from another properties (as pointer) value.
                                        */
        void                            setValue(T* val);


                                        /**
                                            A conversion operator.
                                        */
                                        operator T() const;

                                        /**
                                            A conversion operator.
                                        */
                                        operator T&();


                                        /**
                                            Set a property value from another properties (as reference) value.
                                        */
        void                            setValue(const T& val);

                                        /**
                                            Set a property value from a string.
                                        */
        void                            setValueFromString(const string& val);

                                        /**
                                            Return the value of a property.
                                        */
        T                               getValue() const;

                                        /**
                                            Get a reference to a properties value.
                                        */
        T&                              getValueReference();

                                        /**
                                            Get a pointer to a properties value.
                                        */
        T*                              getValuePointer();

                                        /**
                                            Get a handle (void*) to a properties value.
                                        */
        void*                           getValueHandle();

                                        /**
                                            Get a string representation of a properties value.
                                        */
        string                          getValueAsString() const;

                                        /**
                                            Set a properies value from an opaque pointer
                                        */
        void                            setValue(const void* value);

                                        /**
                                            Clear a property's value
                                        */
        bool                            clearValue();

        PropertyBase*                   getCopy() const;

     protected:
                                        /**
                                            \brief The value of the property.
                                        */
        T                               mValue;

};

template<class T>
Property<T>::Property(  const T& value,         const string& name,
                        const string& hint,     const string& descr,
                        const string& alias,    bool readonly
)
:
PropertyBase(getPropertyType(value), name, hint, descr, alias, readonly),
mValue(value)
{}

template<class T>
Property<T>::~Property()
{}

template<class T>
Property<T>::Property(const Property<T>& para)
:
PropertyBase(para)
{
    mValue = para.mValue;
}


template<class T>
Property<T>& Property<T>::operator=(const Property<T>& rhs)
{
    PropertyBase::operator = (rhs);
    mValue = rhs.mValue;
    return (*this);
}

template<class T>
inline PropertyBase* Property<T>::getCopy() const
{
    return new Property<T>(*this);
}

/**
    Set a property value
*/
template<class T>
void Property<T>::setValue(const T& val)
{
    mValue = val;
}

template<class T>
T Property<T>::getValue() const
{
    return mValue;
}

template<class T>
string Property<T>::getValueAsString() const
{
    return toString(mValue);
}

template<class T>
T* Property<T>::getValuePointer()
{
    return &(mValue);
}

template<class T>
T& Property<T>::getValueReference()
{
    return mValue;
}

template<class T>
void* Property<T>::getValueHandle()
{
    return (void*) &mValue;
}

template<class T>
bool Property<T>::clearValue()
{
    return false;
}
//================= SPECIALIZATIONS ====================

//================= BOOL ===============================
/**
    Set a bool properties value, from a string.
*/
template<>
inline void Property<bool>::setValueFromString(const string& val)
{
    mValue = rr::toBool(val);
}

template<>
inline void Property<bool>::setValue(const void* val)
{
    mValue = * ((bool*) (val));
}

template<>
inline Property<bool>::operator bool () const
{
    return mValue;
}
//================= Int ===============================
/**
    Set an int properties value, from a string.
*/
template<>
inline void Property<int>::setValue(const void* val)
{
    mValue = * ((int*) (val));
}

template<>
inline void Property<int>::setValueFromString(const string& val)
{
    mValue = rr::toInt(val);
}

template<>
inline bool Property<int>::clearValue()
{
    mValue = 0;
    return true;
}

template<>
inline Property<int>::operator int () const
{
    return mValue;
}

template<>
inline Property<int>::operator int & ()
{
    return mValue;
}

//================= Double ===============================
template<>
inline Property<double>::operator double& ()
{
    return mValue;
}

template<>
inline void Property<double>::setValue(const void* val)
{
    mValue = * ((double*) (val));
}

/**
    Set a double properties value, from a string.
*/
template<>
inline void Property<double>::setValueFromString(const string& val)
{
    mValue = rr::toDouble(val);
}

template<>
inline string Property<double>::getValueAsString() const
{
    return rr::toString(mValue, "%G");
}

//================= std::string ===============================

template<>
inline void Property<string>::setValue(const void* val)
{
    mValue = * ((string*) (val));
}

/**
    Set a string properties value, from a string.
*/
template<>
inline void Property<string>::setValueFromString(const string& str)
{
    mValue = str;
}

template<>
inline bool Property<string>::clearValue()
{
    mValue = rr::gEmptyString;
    return true;
}


template<>
inline Property<string>::operator string () const
{
    return mValue;
}

template<>
inline Property<string>::operator string & ()
{
    return mValue;
}

//================= vector<int> ===============================
template<>
inline void Property< vector<int> >::setValue(const void* val)
{
    mValue = * ((vector<int> *) (val));
}

template<>
inline bool Property< vector<int> >::clearValue()
{
    mValue.clear();
    return true;
}

/**
    Set a vector<int> properties value, from a string. This function expects input string
    containing comma delimited values, and braces, e.g. {1, 3, 5 ,2}
*/
template<>
inline void Property< std::vector<int> >::setValueFromString(const string& val)
{
    StringList nrList = rr::splitString(val,", {}");
    mValue.clear();
    for(int i = 0; i < nrList.size(); i++)
    {
        mValue.push_back( toInt(nrList[i]));
    }
}

template<>
inline Property< vector<int> >::operator vector<int> () const
{
    return mValue;
}

//================= vector<double> ===============================
template<>
inline void Property< vector<double> >::setValue(const void* val)
{
    mValue = * ((vector<double> *) (val));
}

/**
    Set a vector<double> properties value, from a string. This function expects input string
    containing comma delimited values, and braces, e.g. {1, 3, 5 ,2}
*/
template<>
inline void Property< std::vector<double> >::setValueFromString(const string& val)
{
    StringList nrList = rr::splitString(val,", {}");
    mValue.clear();
    for(int i = 0; i < nrList.size(); i++)
    {
        mValue.push_back( toDouble(nrList[i]));
    }
}

template<>
inline Property< vector<double> >::operator vector<double> () const
{
    return mValue;
}

//================= vector<string> ===============================
/**
    Set a vector<string> properties value, from a string. This function expects input string
    containing comma delimited values.
*/
template<>
inline void Property< std::vector<string> >::setValueFromString(const string& val)
{
    mValue = rr::splitString(val,", ");
}

//================= rr::StringList ===============================
template<>
inline string Property<rr::StringList>::getValueAsString() const
{
    return mValue.AsString();
}

template<>
inline bool Property<StringList>::clearValue()
{
    mValue = StringList();
    return true;
}

template<>
inline void Property< StringList >::setValue(const void* val)
{
    mValue = * ((StringList* )val);
}

/**
    Set a StringList properties value, from a string. This function expects input string
    containing comma delimited values.
*/
template<>
inline void Property< rr::StringList >::setValueFromString(const string& val)
{
    mValue = rr::splitString(val,", ");
}


//========== Properties container
/**
    Returns a Properties value as a string.
*/
template<>
inline string Property<Properties>::getValueAsString() const
{
    stringstream ss;
    ss << mValue;
    return ss.str();
}

template<>
inline void Property< Properties >::setValue(const void* val)
{
    mValue = * ((Properties* )val);
}

/**
    Set a property value
*/
template<>
inline void Property<Properties>::setValue(Properties* val)
{
    mValue = (*val);
}

/**
    Set a property value
*/
template<>
inline void Property<Properties>::setValue(const Properties& val)
{
    mValue = val;
}

/**
    Set the value of a Properties container value, from a string.
    \note This is not implemented.
*/
template<>
inline void Property<Properties>::setValueFromString(const string& val)
{
    Log(lError)<<"Trying to set Properties container by a string. This is not implemented!";
}

template<>
inline Property<Properties>::operator Properties & ()
{
    return mValue;
}

//===================== Function templates used to retrieve a property's type ===================
/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType(const int& a)
{
    return "int";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType(const bool& a)
{
    return "bool";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType(const double& a)
{
    return "double";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType<string>(const string& a)
{
    return "std::string";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType< vector<int> >(const vector<int> &a)
{
    return "vector<int>";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType< vector<double> >(const vector<double> &a)
{
    return "vector<double>";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType< vector<string> >(const vector<string> &a)
{
    return "vector<string>";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType<rr::StringList>(const rr::StringList& a)
{
    return "stringList";
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType<Properties>(const Properties& value)
{
    return "listOfProperties";
}


//Temporary instantiations here ================

/**
    Set a property value
*/

template<>
inline void Property< ls::Matrix<double> >::setValue(const void* val)
{
    mValue = * ((ls::Matrix<double>* )val);
}

/**
    Set the value of a Properties container value, from a string.
    \note This is not implemented.
*/
template<>
inline void Property< ls::Matrix<double> >::setValueFromString(const string& val)
{
    Log(lError)<<"Trying to set Properties container by a string. This is not implemented!";
}

template<>
inline string getPropertyType< ls::Matrix<double> >(const ls::Matrix<double>& a)
{
    return "matrix";
}

}

#endif

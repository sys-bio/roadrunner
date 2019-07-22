#ifndef telArrayedParameterH
#define telArrayedParameterH
#include <ostream>
#include <string>
#include <sstream>
#include "../../rrIniSection.h"
#include "../../rrIniKey.h"
#include "telProperty.h"
#include "telCommonExporter.h"
#include "../../rrException.h"
//---------------------------------------------------------------------------
namespace rr
{

using std::string;
using std::ostream;
using std::stringstream;

class COMMON_DECLSPEC ArrayedParameter
{

    public:
                                        ArrayedParameter(const string& lbl = gEmptyString, double initialVal = 0, double incr = 0, int nrOfIncrements = 0);

        void                            increment();
        void                            decrement();
        /**
            Output an arrayed parameter as a string to an ostream
        */
        friend ostream&                 operator<<(ostream& os, ArrayedParameter& pm);

        string                          asString() const;
        int                             getNumberOfIncrements() const;
        void                            setNumberOfIncrements(int nr);
        double                          getIncrement() const;
        void                            setIncrement(double incr);
        double                          getCurrentValue() const;
        void                            setCurrentValue(double val);
        string                          getName() const;
        void                            setName(const string& str);
        double                          getInitialValue() const;
        void                            setInitialValue(double val);

    protected:
        string                          mName;
        double                          mInitialValue;
        double                          mIncrement;
        int                             mNumberOfIncrements;
        double                          mCurrentValue;

};

template<>
inline string Property< ArrayedParameter >::getValueAsString() const
{
    return mValue.asString();
}

template<>
inline void Property< ArrayedParameter>::setValueFromString(const string& val)
{
    stringstream msg;
    IniSection section(val,'\n');
    IniKey* key = section.GetKey("Name");
    if(!key)
    {
        msg << "Failed finding key: \"Name\" in setting ArrayedParameter value from string.";
        throw(Exception(msg.str()));
    }
    else
    {
        mValue.setName(key->mValue);
    }

    key = section.GetKey("InitialValue");
    if(!key)
    {
        msg << "Failed finding key: \"InitialValue\" in setting ArrayedParameter value from string.";
        throw(Exception(msg.str()));
    }
    else
    {
        mValue.setInitialValue(toDouble(key->mValue));
    }

    key = section.GetKey("Increment");
    if(!key)
    {
        msg << "Failed finding key: \"Increment\" in setting ArrayedParameter value from string.";
        throw(Exception(msg.str()));
    }
    else
    {
        mValue.setIncrement(toDouble(key->mValue));
    }

    key = section.GetKey("NumberOfIncrements");
    if(!key)
    {
        msg << "Failed finding key: \"NumberOfIncrements\" in setting ArrayedParameter value from string.";
        throw(Exception(msg.str()));
    }
    else
    {
        mValue.setNumberOfIncrements(toInt(key->mValue));
    }

    key = section.GetKey("CurrentValue");
    if(!key)
    {
        msg << "Failed finding key: \"CurrentValue\" in setting ArrayedParameter value from string.";
        throw(Exception(msg.str()));
    }
    else
    {
        mValue.setCurrentValue(toDouble(key->mValue));
    }
}

template<>
inline string getPropertyType< ArrayedParameter >(const ArrayedParameter& value)
{
    return "arrayedParameter";
}

}
#endif

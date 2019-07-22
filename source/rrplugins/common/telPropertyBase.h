/**
 * @file telPropertyBase.h
 * @brief RoadRunner Property class implementation
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
#ifndef telPropertyBaseH
#define telPropertyBaseH
#include <string>
#include <ostream>
#include "../../rrConstants.h"
#include "telCommonExporter.h"
//---------------------------------------------------------------------------

namespace rr
{

using std::string;
using std::ostream;

/**
 * \brief The PluginProperty class is an interface for derived Properties, that can be of any type.
 * A property has a name, description and a hint, encapsulating the the purpose of the actual property.
 */
class COMMON_DECLSPEC PropertyBase
{
    public:
    //Friends first
        /**
         * Ouput the property to an output stream.
         */
        COMMON_DECLSPEC
        friend ostream&                     operator<<(ostream& stream, const PropertyBase& outMe);

    public:
        /**
         * The constructor populates the parameters name and hint.
         */
                                            PropertyBase(   const string& type, const string& name,
                                                            const string& hint, const string& descr,
                                                            const string& alias = rr::gEmptyString, bool  readOnly = false);
        /**
         * De-allocate any memory allocated.
         */
        virtual                            ~PropertyBase();

        /**
            Copy constructor. 
        */            
                                            PropertyBase(const PropertyBase& pp);
                                            
        /**
            Assignment operator. This creates a deep copy of the object on the right side of the equal sign.
        */            
        PropertyBase&                       operator=(const PropertyBase& rhs);

        virtual PropertyBase*               getCopy() const = 0;

        /**
         * Return the parameters name
         */
        string                              getName() const;

        /**
         * Return the alias for the property, if any
         */
        string                              getAlias() const;

        /**
         * Set a properties alias
         */
        bool                                setName(const string& alias);

        /**
         * Return description of the property
         */
        string                              getDescription() const;

        /**
         * Set the description of a property
         */
        void                                setDescription(const string& descr);

        /**
         * Set the hint of a property
         */
        void                                setHint(const string& hint);

        /**
         * Return the parameters hint
         */
        string                              getHint() const;

        /**
         * Return the property in a string representation
         */
        string                              asString() const;

        /**
         * Get the type of the property.
         */
        string                              getType() const;

        /**
         * Is the property intended to be read only? OBSERVE: The read only property is a hint to a client
            that the intention of the property is to be read only. A client can STILL change the property value.
         */
        bool                                isReadOnly() const;

        //Virtuals
        /**
         *  Get the value of the property, as a string.
         */
        virtual string                      getValueAsString() const;

        /**
         * Get an opaque pointer to the value of the property.
         */
        virtual void*                       getValueHandle();

        /**
         * Set the value of the property, using string representation.
         */
        virtual void                        setValue(const void* value) = 0;

        /**
         * Set the value of the property, using string representation.
         */
        virtual void                        setValueFromString(const string& value) = 0;

        /**
         * /brief Clear the property data. This function is implemented in descendants.
         */
        virtual bool                        clearValue() = 0;



    protected:
        /**
         * The name of the property.
         */
        string                              mName;

        /**
         * A property may have one alias
         */
        string                              mAlias;

        /**
         * A property may be read only
         */
        bool                                mReadOnly;

        /**
         * The description of a property.
         */
        string                              mDescription;

        /**
         * The hint of the property.
         */
        string                              mHint;

        /**
         * The type of the property.
         */
        string                              mType;
};

}
#endif


/*
 * Variant.h
 *
 *  Created on: Apr 26, 2014
 *      Author: andy
 */

#ifndef VARIANT_H_
#define VARIANT_H_

#include "rrExporter.h"
#include <typeinfo>
#include <string>

namespace rr
{

/**
 * A basic type to hold a variety of data types.
 *
 * Unfourtunatly C++ does not have a standard variant type so we have to
 * create one here.
 *
 * This is a fairly low performance class and is intended soley for
 * infrequent operations such as setting configuration parameters.
 *
 * If Python support is enabled, this class can convert to and from a
 * Python object.
 *
 * Rationale:
 * C++ does not have a built in variant type. Other variant types exist
 * such as boost::any and Poco dynamic var. However including the one of these
 * as part of our public API would forever tie us to that particular library, and
 * would impose yet another dependency.
 *
 * Only very basic variant type support is needed and this class exposes the
 * absolute minimum possible API to support such features, and provides
 * just two methods of getting and storing native C++ types in it.
 *
 * This is *currently* implemented by Poco but the implementation is fully
 * opaque and may change in the future.
 */
class RR_DECLSPEC Variant
{
public:

    /**
     * creates an emtpy variant
     */
    Variant();

    /**
     * create a new variant from an existing supported data type.
     */
    template <typename T>
    Variant(const T& val) : self(0)
    {
        const std::type_info &info = typeid(T);
        alloc();
        assign(info, &val);
    }

    /**
     * Copy constructor.
     */
    Variant(const Variant& other);

    /**
     * Assignment operator for assigning POD to Var
     */
    template <typename T>
    Variant& operator = (const T& value)
    {
        const std::type_info &info = typeid(T);
        assign(info, &value);
        return *this;
    }

    /**
     * Assignment operator
     */
    Variant& operator = (const Variant& other);

    virtual ~Variant();

    const std::type_info& type() const;

    /**
     * convert this variant to a supported data type.
     */
    template <typename T>
    T convert() const
    {
        T value;
        convert_to(typeid(T), &value);
        return value;
    }

    /**
     * Parses the string which must be in JSON format
     */
    static Variant parse(const std::string& val);

    /**
     * Converts the Variant to a string in JSON format.
     */
    std::string toString() const;

    bool isString() const;

    bool isInteger() const;

    bool isNumeric() const;

    bool isBool() const;

    /**
     * true if empty.
     */
    bool isEmpty() const;

    /**
     * true if this is a signed number.
     */
    bool isSigned() const;





private:
    /**
     * private implementation, this gives us the flexibility to use
     * different implementations in the future and maintain binary
     * compatability.
     */
    struct VariantImpl *self;
    void alloc();
    void assign(const std::type_info& info, const void* value);
    void convert_to(const std::type_info& info, void* value) const;
};

} /* namespace rr */

#endif /* VARIANT_H_ */

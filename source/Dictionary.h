/*
 * Dictionary.h
 *
 *  Created on: Jul 14, 2014
 *      Author: andy
 */

#ifndef DICTIONARY_H_
#define DICTIONARY_H_

#include "Setting.h"
#include <string>
#include <vector>
#include <unordered_map>
#include "rrExporter.h"

#if defined(SWIG) || defined(SWIGPYTHON)
#include "PyUtils.h"
#endif


namespace rr {

/**
 * @brief This class is frozen, no new features
 * A dictionary interface that objects can implement. These are automatically
 * mapped to python dictionaries.
 */
class RR_DECLSPEC Dictionary  // Yes, I know this is a pure virtual *INTERFACE*
{                             // so it should NOT need to be exported, but MSVC
public:                       // *mistakenly* complains in ONE instance.
                              // add the declspec just to shut it up.

#ifndef SWIG

    /**
     * Set the value for an arbitrary key.
     *
     * The Setting object has a large number of creation operators so
     * you only have to give the value as native C++ object i.e.
     *
     * @code
     * Dictionary*d = (...);
     * d->setItem("someIntKey", 1);
     * d->setItem("someStrKey", std::string("some std::string"));
     * d->setItem("someFloatVal, 0.123);
     * @endcode
     */
    virtual void setItem(const std::string& key, const rr::Setting& value) = 0;

    /**
     * Get a value. Variants are POD. If the key does not exist, an std::exception
     * will be thrown.
     *
     * The Setting object has the assigment operator overloaded so
     * it will automatically convert to any native C++ object. If the
     * Setting can not be converted, it will raise an exception, i.e.
     *
     * @code
     * Dictionary* d = ...;
     * int a = d->getItem("someIntKey");
     * std::string s = d->getItem("someStrVal");
     * double val = d->getItem("someDoubleVal");
     * @endcode
     */
    virtual Setting getItem(const std::string& key) const = 0;

    /**
     * is there a key matching this name.
     */
    virtual bool hasKey(const std::string& key) const = 0;

    /**
     * remove a value
     */
    virtual size_t deleteItem(const std::string& key) = 0;

    /**
     * list of keys in this object.
     */
    virtual std::vector<std::string> getKeys() const = 0;

    /**
     * Pure virtual interface, you should never have to delete
     * an instance of this type directly.
     */
    virtual ~Dictionary() {};

#endif

    /**
     * implement the python method if we are being used in the python binding.
     *
     * Note, if these methods are added to an %extend section, derived classes
     * will not see them, so as far as I can tell, this is the only way to
     * add these methods.
     */
#if defined(SWIGPYTHON)
    PyObject *keys() {
        return dictionary_keys(this);
    }

    PyObject *values() {
        return dictionary_values(this);
    }

    PyObject *items() {
        return dictionary_items(this);
    }

    PyObject *__getitem__(const char* key) {
        return dictionary_getitem(this, key);
    }

    PyObject *__setitem__(const char* key, PyObject* value) {
        return dictionary_setitem(this, key, value);
    }

    void __delitem__(const char* key) {
        return dictionary_delitem(this, key);
    }

    PyObject *__contains__(const char* key) {
        return dictionary_contains(this, key);
    }

    std::string helloPython() {
        return "hello python";
    }
#endif
};


/**
 * @brief This class is frozen, no new features
 * Basic implementation of the Dictionary interface which uses a
 * std unordered std::map to store the values.
 */
class RR_DECLSPEC BasicDictionary : public Dictionary
{
public:
    BasicDictionary() = default;

    /**
     * sets an item in the internal unordered std::map.
     */
    void setItem(const std::string& key, const rr::Setting& value) override;

    /**
     * gets an item from the internal unordered std::map.
     */
    Setting getItem(const std::string& key) const override;

    /**
     * is there a key matching this name.
     *
     * @retruns true if this key exists, false otherwise.
     */
    bool hasKey(const std::string& key) const override;

    /**
     * remove a value
     */
    size_t deleteItem(const std::string& key) override;

    /**
     * list of keys in this object.
     */
    std::vector<std::string> getKeys() const override;

    /**
     * Pure virtual interface, you should never have to delete
     * an instance of this type directly.
     */
    ~BasicDictionary() override = default;

protected:
    typedef std::unordered_map<std::string, Setting> VariantMap;
    VariantMap items;
};

}





#endif /* DICTIONARY_H_ */

/*
 * Dictionary.h
 *
 *  Created on: Jul 14, 2014
 *      Author: andy
 */

#ifndef DICTIONARY_H_
#define DICTIONARY_H_

#include "Variant.h"
#include <string>
#include <vector>


namespace rr {

/**
 * A dictionary interface that objects can implement. These are automatically
 * mapped to python dictionaries.
 */
class RR_DECLSPEC Dictionary  // Yes, I know this is a pure virtual *INTERFACE*
{                             // so it should NOT need to be exported, but MSVC
public:                       // *mistakenly* complains in ONE instance.
                              // add the declspec just to shut it up.

    /**
     * set an arbitrary key
     */
    virtual void setValue(const std::string& key, const rr::Variant& value) = 0;

    /**
     * get a value. Variants are POD.
     */
    virtual Variant getValue(const std::string& key) const = 0;

    /**
     * is there a key matching this name.
     */
    virtual bool hasKey(const std::string& key) const = 0;

    /**
     * remove a value
     */
    virtual int deleteValue(const std::string& key) = 0;

    /**
     * list of keys in this object.
     */
    virtual std::vector<std::string> getKeys() const = 0;

    /**
     * Pure virtual interface, you should never have to delete
     * an instance of this type directly.
     */
    virtual ~Dictionary() {};
};

}





#endif /* DICTIONARY_H_ */
